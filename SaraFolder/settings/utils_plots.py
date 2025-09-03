import os

from matplotlib import pyplot as plt

from SaraFolder.settings import running_settings


def plot_training_history(history, title):
    # Extract training & validation metrics
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(loss) + 1)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    # Plot accuracy
    ax1.plot(epochs, acc, 'o-', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'o-', label='Validation Accuracy')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot loss
    ax2.plot(epochs, loss, 'o-', label='Training Loss')
    ax2.plot(epochs, val_loss, 'o-', label='Validation Loss')
    ax2.set_title('Training vs Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(running_settings.figures_path + os.sep + title, dpi=400)
    plt.show()