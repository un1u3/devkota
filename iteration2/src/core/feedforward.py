import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class FeedForward(nn.Module):
    # feed forward using SWiGLU activation
    # tried other but it seems bestfit
    #  swiGLU(x) = Swish(xW).(xV)
    # where Swish(x) = x * sigmoid(x)
    # 

    def __init__(self,d_model, d_ff = None, dropout = 0.1):
        super().__init__()

        if d_ff is None:
            # swiGLU typically uses a smaller d_FFF
            d_ff = int(2.67 * d_model) 
        
        self.d_model = d_model
        self.d_ff = d_ff

        # threee linear transformations for SwiGLU 
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

        self.dropout = nn.Dropout(p = dropout)

    def forward(self,x):
        swish_out = F.silu(self.w1(x))
        gate_out = self.w3(x)
        x = swish_out * gate_out
        x = self.dropout(x)

        # project back to d_model
        x = self.w2(x)
        x = self.dropout(x)
        return x 

        
