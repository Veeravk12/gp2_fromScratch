import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken  #tokenizer for GPT-2
import os

class GPTDataset(Dataset):
    def __init__(self, text_file, tokenizer, max_length=256, stride=128):
        self.input_ids = []
        self.target_ids = []

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        #tokenize entire text
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        #create sequences with a sliding window
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i:i + max_length]
            target_chunk = tokens[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# def create_dataloader(text_file, batch_size=4, max_length=256, stride=128, shuffle=True):
#     tokenizer = tiktoken.get_encoding("gpt2")
#     dataset = GPTDataset(text_file, tokenizer, max_length, stride)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
