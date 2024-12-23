import matplotlib.pyplot as plt


def plot_loss(train_loss_history, val_loss_history):
    epochs = len(train_loss_history)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_history, label="Training Loss", marker='o')
    plt.plot(range(1, epochs + 1), val_loss_history, label="Validation Loss", marker='o')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
