# devkota: a small GPT-2 style model that writes Nepali poems.
#
#   1. python main.py prepare                      # tokenize corpus + poems into data/tokens/*.bin
#   2. python main.py pretrain                     # learn Nepali from the general corpus
#   3. python main.py finetune                     # adapt to Devkota's poems
#   4. python main.py generate --prompt "हिमाल"     # write a poem
import argparse
import torch

from src.data import build_token_file
from src.tokenizer import load
from src.train import train, load_model

TOKENS = "data/tokens"
PRETRAIN_CKPT = "checkpoints/pretrain.pt"
POEM_CKPT = "checkpoints/devkota_poet.pt"


def cmd_prepare(args):
    jobs = [
        ("data/merged/train_merged.txt", f"{TOKENS}/train.bin", "eos"),
        ("data/merged/validation_merged.txt", f"{TOKENS}/val.bin", "eos"),
        ("data/preprocessed/poem.txt", f"{TOKENS}/poem.bin", "newline"),
    ]
    for src, dst, sep in jobs:
        n = build_token_file(src, dst, sep)
        print(f"{src} -> {dst}: {n:,} tokens")


def cmd_pretrain(args):
    config = dict(vocab_size=load().vocab_size(), d_model=args.d_model,
                  num_layers=args.num_layers, num_heads=args.num_heads,
                  max_seq_len=args.max_seq_len, dropout=0.1)
    train(f"{TOKENS}/train.bin", f"{TOKENS}/val.bin", args.out, config=config,
          max_steps=args.steps, batch_size=args.batch_size, lr=args.lr,
          eval_every=args.eval_every, device=args.device)


def cmd_finetune(args):
    # smaller lr and fewer steps: we only nudge the pretrained model toward poetry
    train(f"{TOKENS}/poem.bin", None, args.out, init_from=args.init_from,
          max_steps=args.steps, batch_size=args.batch_size, lr=args.lr,
          warmup=50, eval_every=args.eval_every, device=args.device)


def cmd_generate(args):
    sp = load()
    model, _ = load_model(args.ckpt, args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ids = sp.encode(args.prompt) if args.prompt else [sp.bos_id()]
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    out = model.generate(input_ids, args.max_tokens, temperature=args.temperature,
                         top_k=args.top_k, eos_token_id=sp.eos_id())
    print(sp.decode(out[0].tolist()))


def main():
    p = argparse.ArgumentParser(description="devkota: Nepali poem GPT")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)

    # defaults are sized for a CPU laptop (~6M params); scale up on a GPU
    pt = sub.add_parser("pretrain")
    pt.add_argument("--steps", type=int, default=5000)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--max-seq-len", type=int, default=128)
    pt.add_argument("--d-model", type=int, default=256)
    pt.add_argument("--num-layers", type=int, default=6)
    pt.add_argument("--num-heads", type=int, default=4)
    pt.add_argument("--eval-every", type=int, default=250)
    pt.add_argument("--out", default=PRETRAIN_CKPT)
    pt.set_defaults(fn=cmd_pretrain)

    ft = sub.add_parser("finetune")
    ft.add_argument("--init-from", default=PRETRAIN_CKPT)
    ft.add_argument("--steps", type=int, default=1000)
    ft.add_argument("--batch-size", type=int, default=16)
    ft.add_argument("--lr", type=float, default=1e-4)
    ft.add_argument("--eval-every", type=int, default=100)
    ft.add_argument("--out", default=POEM_CKPT)
    ft.set_defaults(fn=cmd_finetune)

    g = sub.add_parser("generate")
    g.add_argument("--ckpt", default=POEM_CKPT)
    g.add_argument("--prompt", default="")
    g.add_argument("--max-tokens", type=int, default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(fn=cmd_generate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
