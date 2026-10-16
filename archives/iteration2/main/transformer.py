import torch 
import torch.nn as nn 
from src.core.multi_head_attention import MultiHeadAttention, create_casual_mask
from src.core.feedforward import FeedForward



class TransformerBlock(nn.Module):
    # A single transformer decoder block 
    # Arch :  LayerNorm --> self-attn -->residual conncetion 
    #         Layer-> feedforward->residual connecction 

    def __init__(self, d_model, num_heads =8,  d_ff = None, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads 

        # multihead selff attention 

        self.self_attention = MultiHeadAttention(d_model= d_model,num_heads=num_heads,dropout=dropout)
        
        # feed forward network 
        self.feed_forward = FeedForward( d_model=d_model, d_ff= d_ff, dropout= dropout)

        # layer normalizationn
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # dropout for residual connection 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # 1 layer Noem  1 
        attn_input  = self.ln1(x)


        # 2. self attn 
        attn_output = self.self_attention(
            query = attn_input,
            key = attn_input,
            value = attn_input,
            mask  = mask
        )

        # 3. residual connection 
        x = x + self.dropout(attn_output)

        # feed forward with residuaal 
        ff_input = self.ln2(x)

        # feed-forward
        ff_output = self.feed_forward(ff_input)

        x = x + self.dropout(ff_output)
        return x



