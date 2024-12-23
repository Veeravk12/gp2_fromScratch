import os
import tiktoken
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from .gpt_dataset_class import GPTDataset  # Import the correct Dataset class

# Function to create train and validation loaders
def load_and_split_data(text_file, config):
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load the dataset
    dataset = GPTDataset(
        text_file=text_file, 
        tokenizer=tokenizer, 
        max_length=config["context_length"], 
        stride=config["context_length"] // 2  # Use 50% overlap
    )
    
    total_tokens = len(dataset.input_ids) * config["context_length"]
    train_ratio = 0.9  # For example, 90% training, 10% validation
    
    # Sanity check
    if total_tokens * train_ratio < config["context_length"]:
        print("Not enough tokens for the training loader. "
              "Try to lower the `context_length` or increase the `training_ratio`")
    
    if total_tokens * (1 - train_ratio) < config["context_length"]:
        print("Not enough tokens for the validation loader. "
              "Try to lower the `context_length` or decrease the `training_ratio`")
    
    # Split dataset into train and validation sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader

