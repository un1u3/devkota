import torch
import torch.nn as nn 
import math 


class PositionalEncoding(nn.Module):
    # d_model(int): dimension of the model embeddings 
    # max_seq_len : max sequence length to pre-compute
    # droupout : Dropout probablity 


    def __init__(self, d_model, max_seq_len = 512, dropout= 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # create positional encoding matrix 
        # shape (max_seqlenn, d_model)
        
        pe = torch.zeros(max_seq_len, d_model)


        # 
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # ceate div for sinusoidal functions
        # div_term - 1.  (10000^ (2i/d_model)) 
        # using log space exo*2i *  - log(100000)/d_model

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sin to even indices 0,2,4
        pe[:,0::2] = torch.sin(position * div_term)
        # cos to odd
        pe[:,1::2] = torch.cos(position * div_term)
        # add batch dim (1,max_seqlen, dmodel)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # Add positional encoding (broadcasting across batch dimension)
        # self.pe[:,:seq_len] has shape (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)




