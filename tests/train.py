import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import get_scheduler
from src.models.gpt2_model import GPT2Model
from src.create_dataset.data_loading_splitting import load_and_split_data
from src.models.config import GPT_CONFIG_124M

def train_model(model, train_loader, config, device):
    # Print the device being used
    print(f"Training on device: {device}")

    # Ensure the model is on the correct device
    model = model.to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Loss function
    loss_fn = CrossEntropyLoss()

    train_loss_history = []

    # Training loop
    model.train()
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_loss = 0

        for input_ids, target_ids in tqdm(train_loader, desc="Training"):
            # Move data to the correct device (GPU or CPU)
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        print(f"Training Loss: {avg_loss}")

    return train_loss_history, model


def save_model(model, path="gpt2_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
