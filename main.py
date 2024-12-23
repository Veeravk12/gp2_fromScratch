import torch
from src.models.gpt2_model import GPT2Model
from src.create_dataset.data_loading_splitting import load_and_split_data
from src.models.config import GPT_CONFIG_124M
from tests.train import train_model, save_model
from tests.evaluate import validate_model
from tests.visualisation import plot_loss

def main(text_file, model_save_path="gpt2_model.pth", device="cuda"):
    # Load config
    config = GPT_CONFIG_124M

    # Load data
    train_loader, val_loader = load_and_split_data(text_file, config)

    # Check if data is loading correctly
    print(f"Training batches: {len(train_loader)}")  # Check how many batches you have
    print(f"Validation batches: {len(val_loader)}")  # Check how many validation batches you have

    # Initialize model
    print(f"Initializing model...")
    model = GPT2Model().to(device)

    # Train the model
    print(f"Starting training on {device}...")
    train_loss_history, model = train_model(model, train_loader, config, device)

    # Initialize an empty list for validation loss history
    val_loss_history = []

    # Validate the model after each epoch
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']} - Validating...")
        val_loss = validate_model(model, val_loader, device)
        print(f"Validation loss after epoch {epoch + 1}: {val_loss}")
        val_loss_history.append(val_loss)

    # Save the trained model
    print(f"Saving the trained model to {model_save_path}...")
    save_model(model, model_save_path)

    # Visualize training and validation loss
    print(f"Plotting the loss...")
    plot_loss(train_loss_history, val_loss_history)


if __name__ == "__main__":
    text_file = r"D:\projects\llms_from_Scratch\gpt2\data\raw\the-verdict.txt"  # Replace with the path to your dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    main(text_file, device=device)
