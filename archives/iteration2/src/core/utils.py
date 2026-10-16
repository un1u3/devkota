import torch 
import math 
import json 
from pathlib import Path
from datetime import datetime



class LRScheduler:
    def __init__(self, optimizer, peak_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0 

    def get_lr(self):
        # Linear warmup prevents unstable updates at the start of training
        if self.step_count < self.warmup_steps:
            return self.peak_lr * (self.step_count / self.warmup_steps)
        else:
            #  Cosine decay smoothly anneals LR to ~=0 by total_steps  ~=0(it looks like something hahaha)
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
    def step(self):
        self.step_count +=1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr 
        return lr 
     
    # resource ko kami+ model deviate nahos
def save_checkpoint(path, model, optimizer, step, epoch, loss):
    torch.save({
        'step':step,
        'epoch':epoch,
        'model': model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'loss':loss
    }, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint

# this need some work
def compute_preplx(loss):
    return math.exp(loss)
    
