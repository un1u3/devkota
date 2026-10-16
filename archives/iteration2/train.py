import torch
from main.devkota import Devkota
from src.core.dataset import get_dataloaders
from src.core.trainer import Trainer
from src.core.config import ModelConfig, PreTrainConfig
from src.core.train_spm import NepaliTokenizer
import os

def main():
    print("Devkota training script ")
    print("-" * 50)
    
    # make dirs if not exist
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    # load config
    model_cfg = ModelConfig()
    train_cfg = PreTrainConfig()
    
    # init model
    print("initilizing model...")
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
    
    # print params
    # borrowed logic to count params 
    param_count = model.count_parameters()
    print(f"Total params: {param_count['total']}")
    
    # tokenizer path
    tokenizer_path = "tokenizer/devkota_tokenizer.model"
    if not os.path.exists(tokenizer_path):
        print("tokenizer xina .. run train_tokenizer.py first")
        return

    print("loading tokenizer...")
    tokenizer = NepaliTokenizer(tokenizer_path)
    
    # dataloaders
    print("getting dataloaders...")
    train_loader, val_loader = get_dataloaders(
        train_cfg.train_data,
        train_cfg.val_data,
        tokenizer,
        batch_size=train_cfg.batch_size,
        max_len=model_cfg.max_seq_len
    )
    
    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"training on {device}")
    
    # trainer init
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device
    )
    
    # start trainig
    print("starting training loop...")
    trainer.train()

    print("Training done !!")
    print(f"best model at: {train_cfg.checkpoint_dir}/best_model.pt")

if __name__ == "__main__":
    main()
