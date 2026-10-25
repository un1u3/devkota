# layer norm is used to stabilize the distibution of activation during training 
import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-6):
        super().__init__()
        self.eps = eps 

        # normalize eachh token's vector (over d_model) to mean 0,
        # var 1 the let the model rescale(gamme) and shift (beta) it.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True) # (B,T,1)
        var = x.var(dim= -1, keepdim= True, unbiased = False) # (B,T,1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps) # (B,T,d_model)
        return self.gamma * x_hat + self.beta  # (B,T,d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and B2   according to paper
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   #(B, T, d_model)
        h = self.linear1(x) # (B, T, d_ff)
        h = torch.relu(h)   # non-linearity 
        h = self.dropout(h) # regularization  
        return self.linear2(h) # (B, T, d_model)

    # future reference : T = seq_len
    
    
