import math 
import torch 
import torch.nn as nn 
try:
    from .layerNorm import LayerNorm
except ImportError:
    from layerNorm import LayerNorm

class ResidualConnection(nn.Module):  # This is skip conncection in the fig of the paper

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward (self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

    
