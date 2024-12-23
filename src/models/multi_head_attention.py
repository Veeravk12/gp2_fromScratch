# MultiHeadAttention class

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "Embedding dimension(d_out) must be divisible by number of heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) 
        self.dropout = nn.Dropout(dropout)
        
        # Create the causal mask for self-attention
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # Assert that num_tokens is less than or equal to context_length
        assert num_tokens <= self.mask.shape[0], "Sequence length exceeds context length"

        # Linear transformations for queries, keys, and values
        queries = self.W_query(x)  
        keys = self.W_key(x)  
        values = self.W_value(x)  
        
        # Reshape into multi-head dimensions
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2) 
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2) 
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)  
        
        # Apply the causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  
        attn_scores.masked_fill_(mask_bool, float('-inf'))  
        
        # Compute attention weights and apply dropout
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context_vec = (attn_weights @ values).transpose(1, 2) 
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  
        
        # Apply the final projection
        context_vec = self.out_proj(context_vec)  
        
        return context_vec

