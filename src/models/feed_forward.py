# FeedForward class

import torch
import torch.nn as nn
from src.models.gelu import GELU  # Assuming GELU is a custom activation function

class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        
        # Define the FeedForward network using nn.Sequential
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),  # First Linear layer
            GELU(),                           # Activation function
            nn.Linear(4 * emb_dim, emb_dim),  # Second Linear layer
            nn.Dropout(dropout)               # Dropout for regularization
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the sequential layers
