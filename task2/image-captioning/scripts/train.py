import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.captioner import ImageCaptioningModel
from utils.helpers import get_data_loader

# Configure logging
logger = logging.getLogger(__name__)

# Hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
BATCH_SIZE = 32

# Dataset paths
ROOT_DIR = "../dataset"
TRAIN_CAPTIONS = "../dataset/captions.txt"

# Save paths
MODEL_PATH = "../artifacts/models/image_captioning_model.pth"
CHECKPOINT_BASE_PATH = "../artifacts/checkpoints"


def train():
    """Main training function."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load data
    logger.info("Loading data...")
    train_loader, dataset = get_data_loader(
        ROOT_DIR,
        TRAIN_CAPTIONS,
        transform,
        batch_size=BATCH_SIZE
    )

    vocab_size = len(dataset.vocab)
    logger.info("Vocabulary size: %d", vocab_size)

    # Initialize model
    model = ImageCaptioningModel(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=vocab_size,
        num_layers=NUM_LAYERS
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # For logging
    writer = SummaryWriter("runs/exp1")
    step = 0

    # Training loop
    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        epoch_loss = 0

        # Progress bar for batches
        loop = tqdm(train_loader, leave=True)

        for idx, (images, captions) in enumerate(loop):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(images, captions[:, :-1])

            # Calculate loss
            # outputs: (batch_size, caption_length, vocab_size)
            # captions[:, 1:]: target captions (excluding <start> token)
            loss = criterion(
                outputs.reshape(-1, vocab_size),
                captions[:, 1:].reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Log to tensorboard
            writer.add_scalar("Training Loss", loss.item(), global_step=step)
            step += 1

        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': dataset.vocab,
            }
            torch.save(checkpoint, f'{CHECKPOINT_BASE_PATH}/checkpoint_epoch_{epoch + 1}.pth')
            print(f"Checkpoint saved: {CHECKPOINT_BASE_PATH}/checkpoint_epoch_{epoch + 1}.pth")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dataset.vocab,
        'embed_size': EMBED_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
    }, MODEL_PATH)

    logger.info("Training complete! Model saved to '%s'", MODEL_PATH)
    writer.close()

if __name__ == "__main__":
    train()
