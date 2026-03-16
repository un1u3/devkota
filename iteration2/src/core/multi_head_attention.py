import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

class MultiHeadAttention(nn.Module):
    # this  splits the d_model into num_heads ,
    #  performs attention in parallel then concatenates and projects the reslt

    def __init__(self, d_model, num_heads = 8 , dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        #  linear projection for Q, K, V 
        # we use single linear layer, then split into heads 
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # output projection 
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p = dropout)


        # scaling factor 
        self.scale = math.sqrt(self.d_k) 


    def split_heads(self, x):
        # split the last dim into (num_heads, d_k)

        batch_size, seq_len, d_model = x.size()

        # reshape to (batchsize, seqlen, num_heads, dk)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1,2)
    
    def combine_heads(self,x):
        # inverse of split heads 
        batch_size, num_heads, seq_len, d_k = x.size()

        # idk what contiguous does fk it 
        x = x.transpose(1,2).contiguous()
        return x.view(batch_size, seq_len, self.num_heads * d_k)

    def scaled_dot_prod_attn(self, Q, K, V, mask = None):
        # attn score 
        scores = torch.matmul(Q, K.transpose(-2, -1))  / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        
        # applyign softmax to get aattn weights 
        attn_weights = F.softmax(scores, dim=-1)

        # appy dropout 
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # Linear projection 
        Q = self.W_q(query) # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    
        V = self.W_v(value)  
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  
        V = self.split_heads(V)  

        attn_output, _ = self.scaled_dot_prod_attn(Q, K, V, mask)

        # combine heads 
        attn_output = self.combine_heads(attn_output)

        output = self.W_o(attn_output)
        return output
    
def create_casual_mask(seq_len, device):
    # Create a casual (lower triangular) boolean mask for autoregressive gen
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)
    

def create_padding_mask(seq, pad_idx= 3):
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        
    return mask


    
        
