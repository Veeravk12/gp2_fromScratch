import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def validate_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    loss_fn = CrossEntropyLoss()

    with torch.no_grad():  # No gradient computation needed during validation
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

