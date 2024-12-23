import torch
import torch.nn as nn
from src.models.transformer_block import TransformerBlock
from src.models.layer_norm import LayerNorm
from src.models.config import GPT_CONFIG_124M  # Import the config directly


class GPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Access configuration directly from GPT_CONFIG_124M
        config = GPT_CONFIG_124M  # Access configuration directly

        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        # Use nn.Sequential to stack the transformer blocks as per the configuration
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(config["n_layers"])]  # Stack blocks sequentially
        )
        
        # Final LayerNorm and output projection layer
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        # Get token and position embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device).unsqueeze(0).expand(batch_size, -1))  # Corrected position embedding

        # Add token and position embeddings together
        x = tok_embeds + pos_embeds  # Shape: [batch_size, num_tokens, emb_size]
        
        # Apply dropout on the embeddings
        x = self.drop_emb(x)
        
        # Pass through all transformer blocks sequentially
        x = self.trf_blocks(x)
        
        # Apply the final LayerNorm
        x = self.final_norm(x)
        
        # Generate logits via the output head (projection layer)
        logits = self.out_head(x)
        
        return logits
