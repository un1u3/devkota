import torch
import torch.nn as nn
try:
    from .layerNorm import LayerNorm, FeedForward
    from .mha import MultiHeadAttention
except ImportError:
    from layerNorm import LayerNorm, FeedForward
    from mha import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self,d_model, h, d_ff, dropout):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, h, dropout)
        self.ln2  = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # sub-layer 1: attn
        x = x + self.drop(self.attn(self.ln1(x), mask))

        # ssub layer 2
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x
