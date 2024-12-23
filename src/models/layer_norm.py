# LayerNorm class

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

#builin class 
# import torch
# import torch.nn as nn

# class LayerNorm(nn.Module):
#     def __init__(self, emb_dim, eps=1e-5):
#         super().__init__()
#         self.norm = nn.LayerNorm(emb_dim, eps=eps)  # Use built-in LayerNorm

#     def forward(self, x):
#         return self.norm(x)  # Forward pass through built-in LayerNorm

