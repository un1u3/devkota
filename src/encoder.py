import torch 
import torch.nn as nn 
from layerNorm import FeedForward, LayerNorm
from mha import MultiHeadAttention
from residualConnection import ResidualConnection

class EncoderBlock (nn.Module);
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout() for _ in range(2))])

    def forward (self, x, src_mask): #we apply mask to now interact padding word with other words 
        x = self.residual_connections[0] (x, lambda x:self.self_attention_block(x,x,x,src_mask)) #basically lambda function is calling the forward method of multiheadattention
        x = self.residual_connections[1] (x, self.feed_forward_block)
        return x


class Encoder (nn.Module):
    # Paper says we can have n number of encoder blocks

    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)