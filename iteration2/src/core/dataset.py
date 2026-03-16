import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sequences = []
        
        count = 0 
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # stop if max_samples reached (for ram saving)
                if max_samples is not None and count >= max_samples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # encoding line
                tokens = tokenizer.encode(line, add_bos=True, add_eos=True)
                
                # truncate if too long 
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                
                # skip small lines 
                if len(tokens) > 10: 
                    self.sequences.append(tokens)
                    count += 1
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Input and target (shifted by 1)
        input_ids = seq[:-1]
        labels = seq[1:]
        
        # Pad
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [self.tokenizer.pad_id] * pad_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
def get_dataloaders(train_path, val_path, tokenizer, batch_size=8, max_len=512, max_samples=None):
    # max_samples : for limiting lines (ram issues)
    train_dataset = TextDataset(train_path, tokenizer, max_len, max_samples)
    val_dataset = TextDataset(val_path, tokenizer, max_len, max_samples) 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
