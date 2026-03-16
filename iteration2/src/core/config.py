class ModelConfig:
    # 8GB RAM Optimized Config
    vocab_size = 16000     
    d_model = 256          # Reduced from 512
    num_layers = 6         # Reduced from 12
    num_heads = 4          # Reduced from 8
    d_ff = 1024            # Reduced from 2048
    max_seq_len = 128      # Reduced from 512 for faster training
    dropout = 0.1
    pad_idx = 3

class PreTrainConfig:
    # Use smaller subset data
    train_data = "preprocessed_data/train_small.txt"
    val_data = "preprocessed_data/val_small.txt"
    
    batch_size = 4         # Small batch size for 8GB RAM
    accumulation_steps = 8  # Increased to maintain effective batch size
    epochs = 5
    
    lr = 3e-4
    weight_decay = 0.01
    warmup_steps = 1000
    max_grad_norm = 1.0
    
    checkpoint_dir = "checkpoints"
    save_every = 500
    eval_every = 200

class FineTuneConfig:
    # Same changes here
    train_data = "preprocessed_data/devkota_train.txt"
    val_data = "preprocessed_data/devkota_val.txt"
    pretrained = "checkpoints/best_model.pt"
    
    batch_size = 2
    accumulation_steps = 4
    epochs = 10
    
    lr = 1e-5
    weight_decay = 0.01
    warmup_steps = 100
    max_grad_norm = 1.0
    
    patience = 3
    checkpoint_dir = "checkpoints/finetuned"
    save_every = 200
    eval_every = 100
    output = "checkpoints/devkota_poet.pt"