# TransformerBlock class

import torch
import torch.nn as nn
from src.models.multi_head_attention import MultiHeadAttention
from src.models.feed_forward import FeedForward
from src.models.layer_norm import LayerNorm
from src.models.config import GPT_CONFIG_124M

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # Access configuration from GPT_CONFIG_124M
        self.att = MultiHeadAttention(
            d_in=GPT_CONFIG_124M["emb_dim"],
            d_out=GPT_CONFIG_124M["emb_dim"],
            context_length=GPT_CONFIG_124M["context_length"],
            num_heads=GPT_CONFIG_124M["n_heads"], 
            dropout=GPT_CONFIG_124M["drop_rate"],
            qkv_bias=GPT_CONFIG_124M["qkv_bias"]
        )

        # FeedForward block (pass config as necessary)
        self.ff = FeedForward(GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["drop_rate"])

        # LayerNorm for both attention and feed-forward
        self.norm1 = LayerNorm(GPT_CONFIG_124M["emb_dim"])
        self.norm2 = LayerNorm(GPT_CONFIG_124M["emb_dim"])

        # Dropout layers for residual connections
        self.drop_shortcut = nn.Dropout(GPT_CONFIG_124M["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)  # LayerNorm first
        x = self.att(x)  # Pass through attention layer
        x = self.drop_shortcut(x)  # Apply dropout
        x = x + shortcut  # Add original input (residual connection)

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)  # Apply LayerNorm
        x = self.ff(x)  # Pass through feed-forward network
        x = self.drop_shortcut(x)  # Apply dropout
        x = x + shortcut  # Add residual connection
        
        return x
