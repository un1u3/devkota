import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from src.core.positionalencoder import PositionalEncoding
from src.core.multi_head_attention import MultiHeadAttention, create_casual_mask, create_padding_mask
from main.transformer import TransformerBlock


class Devkota(nn.Module):
    # architecture
    # 1.TOken embedding 
    # 2.positional encoding 
    # 3. 12 transformer blocks 
    # 4.Finallayer norm 
    # lang model head (projectt to vocab)


    # ig it needs some value
    def __init__(self, vocab_size, d_model,num_layers, num_heads,d_ff, max_seq_len, dropout, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        # token embedding layer 
        # adds postion info to embedding 
        self.token_embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = d_model,
            padding_idx = pad_idx
        )

        # positional encoding 
        # adds position informatoin to embedddings 
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,

            max_seq_len= max_seq_len,
            dropout= dropout
        )

        # stack of transformer block 
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads= num_heads,
                d_ff= d_ff,
                dropout= dropout) for _ in range(num_layers)
        ])

        # final layer normalization 
        self.ln_f = nn.LayerNorm(d_model)

        # langauge model head 
        # projects hidden states to vocab logits 
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight 

        # initialize weights 
        self.apply(self._init_weights)

        # special scaled initn f residual projections 
        for name, p in self.named_parameters():
            if name.endswith('W_o.weight') or name.endswith('linear2.weight'):
                # scale by 1/sqrt*2*num_layers) for residual connections 
                nn.init.normal_(p, mean=0.0, std=0.2 / (2 * num_layers)**0.5)

    def _init_weights(self, module):
        # initilizing weights using kaiming initialization 
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                # zero out padding embedding
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids, targets=None, return_loss=True):
        batch_size, seq_len = input_ids.shape 

        # create casual mask 
        casual_mask = create_casual_mask(seq_len, device=input_ids.device)


        # create paddding mask 
        padding_mask = create_padding_mask(input_ids, pad_idx=self.pad_idx)

        # combine masks (elemner wise and )
        mask = casual_mask & padding_mask 

        # token embbbedding 
        x = self.token_embedding(input_ids)
        
        # scale embeddings by sqrt(d_model) 
        x = x * (self.d_model ** 0.5)

        # add positional encoding 
        x = self.positional_encoding(x)

        # pass through transformer blocks 
        for block in self.transformer_blocks:
            x = block(x, mask = mask)

        # final layer norm 
        x = self.ln_f(x)

        # project to vocabulary 
        logits = self.lm_head(x)

        # computer loss if targets provided 
        loss = None
        if targets is not None and return_loss:
            # reshaep for cross entropy 
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_idx
            )
        return {
            'logits': logits,
            'loss': loss
        }
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature = 1.0, top_k= None, top_p=None, eos_token_id = 2, repetition_penalty=1.0):
        # generate text 
        # input ids : starting token Ids of shape 
        # max_new_tokens: max num of tokens to generate 
        # top_k: keep only k tokens for samplint 
        # top_p : Nucleus sampling 
        # eos_token_id : end of sequence tokenid 
        # returns token ids


        self.eval()
        for _ in range(max_new_tokens):
            # compherned using gpt so this line is quacky, need to update it 
            # idk when but,keep it until it works
            input_ids_crop = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # forward pass 
            outputs = self.forward(input_ids_crop, targets=None, return_loss=False)
            logits = outputs['logits']

            # get logits for last positio 
            logits = logits[:, -1, :] / temperature

            # simple repetition penalty to reduce verbatim loops
            if repetition_penalty != 1.0:
                for b in range(logits.size(0)):
                    seen = input_ids[b].unique()
                    logits[b, seen] = logits[b, seen] / repetition_penalty

            # apply top-k filtering 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # apply top-p filtering 
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending = True)
                cumlative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # remove tokens with cumulative probablity about the threshold 
                sorted_indices_to_remove = cumlative_probs > top_p 
                # shift the indices to the right to keep the first token above the threshold 
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token generated for all sequences
            if (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def count_parameters(self):
        # useed gpt to make code concise 
        # will remoove this if it wont work 
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Breakdown by component
        embedding_params = sum(p.numel() for p in self.token_embedding.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_blocks.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable,
            'embedding': embedding_params,
            'transformer_blocks': transformer_params,
        }
            




    
        
