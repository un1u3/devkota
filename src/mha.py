import math 
import torch 
import torch.nn as nn 


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h = h 
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        self.w_o = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask = None):
        B, T, _ = x.shape 

        # project then split into head ( b,t,d_model)
        q = self.w_q(x).view(B, T, self.h,self.d_k).transpose(1,2)
        k = self.w_k(x).view(B, T, self.h,self.d_k).transpose(1,2)
        v = self.w_v(x).view(B, T, self.h,self.d_k).transpose(1,2)

        # scaled scores ( B, h, T, d_k) @ (B< h, d_k, T) === (B, h, T,, ,T)
        scores = q @ k.transpose(-2,-1)/ math.sqrt(self.d_k)

        # causal mask :  future postions get infinity, so softmax gives them 0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmax over the last dim, each row sums to 1 
        attn = torch.softmax(scores, dim =-1)
        attn = self.dropout(attn)
        self.attn_weights = attn 

        # weighted sum of values; (B, h, T, T) @ (B, h, T, d_k) -> (B, h, T, d_k)
        out = attn @ v 

        # merge heads (B, h, T, d_k) -> (B, T, d_model)
        out = out.transpose(1,2).contiguous().view(B, T, self.d_model)

        # output projection 
        return self.w_o(out)


def causal_mask(T):
    return torch.tril(torch.ones(T, T)).view(1, 1, T, T)



# Each head will see the full part of sentence but only a small part of embeddings