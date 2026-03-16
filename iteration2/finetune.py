import os
import torch
from main.devkota import Devkota
from src.core.dataset import get_dataloaders
from src.core.trainer import FineTuner
from src.core.config import ModelConfig, FineTuneConfig
from src.core.train_spm import NepaliTokenizer


def main():
    print("Devkota POeLM fine-tune")
    print("-" * 50)

    model_cfg = ModelConfig()
    tune_cfg = FineTuneConfig()

    # sanity check for data
    missing = [p for p in [tune_cfg.train_data, tune_cfg.val_data] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"file xina : {p}")
        return

    # init model
    print("init model...")
    model = Devkota(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
        pad_idx=model_cfg.pad_idx
    )

    # load tokenizer
    tokenizer_path = "tokenizer/devkota_tokenizer.model"
    if not os.path.exists(tokenizer_path):
        print("tokenizer missing, run train_tokenizer.py")
        return
    tokenizer = NepaliTokenizer(tokenizer_path)

    # data
    train_loader, val_loader = get_dataloaders(
        tune_cfg.train_data,
        tune_cfg.val_data,
        tokenizer,
        batch_size=tune_cfg.batch_size,
        max_len=model_cfg.max_seq_len
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"fine-tuning on {device}")

    trainer = FineTuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=tune_cfg,
        device=device
    )

    trainer.train()
    print(f"poem model saved at: {tune_cfg.output}")


if __name__ == "__main__":
    main()
