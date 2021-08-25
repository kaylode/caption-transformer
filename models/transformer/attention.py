import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    Calculate attention
    :input:
        q:          query
        k:          key
        v:          value
        d_k:        scaled term
        mask:       whether to use masking attention
        dropout:    dropout rate
    :output:
    """

    # Query, Key matrix multiplication
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    # If mask, use masking attetion
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax for scaling in range [0,1]
    scores = F.softmax(scores, dim=-1)
    
    # Dropout
    if dropout is not None:
        scores = dropout(scores)

    # Score, Value matrix multiplication
    output = torch.matmul(scores, v)
    return output, scores

class MultiHeadAttention(nn.Module):
    """
    Calculate multihead attention with num_heads
    :input:
        heads:          number of attention heads
        d_model:        embedding dim
        dropout:        dropout rate
    :output:
    """
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        # For visualization
        self.attn = None
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention 
        scores, self.attn = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FastAttention(nn.Module):
    """
    https://github.com/lucidrains/fast-transformer-pytorch
    """
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting queries to query attention logits
        self.to_k_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting keys to key attention logits

        self.to_r = nn.Linear(dim_head, dim_head)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        # calculate query attention logits

        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim = -1)

        # calculate global query token

        global_q = einsum('b h n, b h n d -> b h d', q_attn, q)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        k = k * global_q

        # now calculate key attention logits

        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim = -1)

        # calculate global key token

        global_k = einsum('b h n, b h n d -> b h d', k_attn, k)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # bias the values

        v = v * global_k
        r = self.to_r(v)

        r = r + q # paper says to add the queries as a residual

        # aggregate

        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)