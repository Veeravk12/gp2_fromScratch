GPT_CONFIG_124M = {
    # Model Configuration
    "vocab_size": 50257,           # Size of vocabulary (tokenized)
    "context_length": 256,        # Maximum sequence length (context window)
    "emb_dim": 768,                # Embedding dimension (vector size for each token)
    "n_heads": 12,                 # Number of attention heads in the MultiHeadAttention
    "n_layers": 12,                # Number of Transformer layers
    "drop_rate": 0.1,              # Dropout rate for regularization
    "qkv_bias": False,             # Whether to add bias to the Q, K, V linear projections
    
    # Training Configuration
    "batch_size":4 ,               # Batch size for training (adjust as needed)
    "learning_rate": 1e-4,         # Learning rate for the optimizer
    "weight_decay": 0.001,          # Weight decay for regularization
    "epochs": 10                   # Number of training epochs
}
