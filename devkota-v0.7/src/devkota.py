"""Decoder-only Devkota language model and command-line interface.

Run from the repository root with ``python src/devkota.py info --vocab-size N``
or use ``python -m src.devkota`` when ``src`` is an importable package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Package import: python -m src.devkota
    from .decoderBlock import DecoderBlock
    from .embedding import InputEmbeddings, PositionalEncoding
    from .layerNorm import FeedForward, LayerNorm
    from .mha import MultiHeadAttention, causal_mask
    from .residualConnection import ResidualConnection
    from .tokenizer import load, train
except ImportError:  # Direct script: python src/devkota.py
    from decoderBlock import DecoderBlock
    from embedding import InputEmbeddings, PositionalEncoding
    from layerNorm import FeedForward, LayerNorm
    from mha import MultiHeadAttention, causal_mask
    from residualConnection import ResidualConnection
    from tokenizer import load, train


def create_padding_mask(input_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Mask padding tokens as attention keys, with shape (batch, 1, 1, seq)."""
    return input_ids.ne(pad_idx)[:, None, None, :]


class Devkota(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        if vocab_size <= 0 or num_layers <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, num_layers, and max_seq_len must be positive")
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        # saved in checkpoints so a model can be rebuilt without repeating its size
        self.config = dict(
            vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            d_ff=d_ff, max_seq_len=max_seq_len, dropout=dropout, pad_idx=pad_idx,
        )
        self.token_embedding = InputEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.transformer_blocks = nn.ModuleList(
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )
        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embedding.weight
        self.apply(self._init_weights)
        # gpt-2: scale down the projections that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("attn.w_o.weight") or name.endswith("ffn.linear2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * num_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # gpt-2 init; pytorch's default N(0, 1) embedding is far too large once tied to lm_head
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> dict[str, Optional[torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.size(1) == 0 or input_ids.size(1) > self.max_seq_len:
            raise ValueError("input sequence length must be between 1 and max_seq_len")
        mask = causal_mask(input_ids.size(1)).to(input_ids.device)
        mask = mask & create_padding_mask(input_ids, self.pad_idx)
        x = self.positional_encoding(self.token_embedding(input_ids))
        for block in self.transformer_blocks:
            x = block(x, mask)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None and return_loss:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=self.pad_idx,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.max_seq_len :]
            logits = self(context, return_loss=False)["logits"][:, -1] / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be positive")
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return input_ids

    def count_parameters(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--pad-idx", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")


def _build_model_from_args(args: argparse.Namespace) -> Devkota:
    model = Devkota(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=0.0,
        pad_idx=args.pad_idx,
    )
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint must contain a state dictionary")
        model.load_state_dict(state_dict)
    return model.to(args.device)


def _parse_input_ids(value: str) -> torch.Tensor:
    try:
        ids = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError("--input-ids must be a comma-separated list of integers") from exc
    if not ids:
        raise ValueError("--input-ids must contain at least one token id")
    return torch.tensor([ids], dtype=torch.long)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Devkota decoder-only transformer CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="show model parameter counts")
    _add_model_arguments(info)
    generate = commands.add_parser("generate", help="generate token ids from token ids")
    _add_model_arguments(generate)
    generate.add_argument("--input-ids", required=True, help="comma-separated prompt token ids")
    generate.add_argument("--max-new-tokens", type=int, default=50)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--eos-token-id", type=int)
    tokenizer = commands.add_parser("train-tokenizer", help="train a SentencePiece tokenizer")
    tokenizer.add_argument("--input-file", type=Path, required=True)
    tokenizer.add_argument("--model-prefix", default="devkota_tokenizer")
    tokenizer.add_argument("--tokenizer-vocab-size", type=int, default=8000)
    tokenizer.add_argument("--character-coverage", type=float, default=0.9999)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_cli().parse_args(argv)
    if args.command == "train-tokenizer":
        train(
            input_file=args.input_file,
            model_prefix=args.model_prefix,
            vocab_size=args.tokenizer_vocab_size,
            character_coverage=args.character_coverage,
        )
        return 0
    model = _build_model_from_args(args)
    if args.command == "info":
        print(json.dumps(model.count_parameters(), indent=2))
        return 0
    input_ids = _parse_input_ids(args.input_ids).to(args.device)
    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=args.eos_token_id,
    )
    print(" ".join(str(token_id) for token_id in generated[0].tolist()))
    return 0


__all__ = [
    "Devkota", "DecoderBlock", "FeedForward", "InputEmbeddings", "LayerNorm",
    "MultiHeadAttention", "PositionalEncoding", "ResidualConnection", "causal_mask",
    "create_padding_mask", "load", "main", "train",
]


if __name__ == "__main__":
    raise SystemExit(main())