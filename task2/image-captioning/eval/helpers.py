import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.datasets import FlickrDataset
from utils.captions import CaptionCollate
from models.captioner import ImageCaptioningModel

def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab = checkpoint['vocab']
    embed_size = checkpoint['embed_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    vocab_size = len(vocab)

    # Initialize model
    model = ImageCaptioningModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, vocab


def generate_caption(image_path, model, vocab, device, max_length=20):
    """
    Generate caption for a single image.
    """
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        caption_indices = model.generate_caption(image_tensor, max_length)

    # Convert indices to words
    caption_words = []
    for idx in caption_indices:
        word = vocab.itos[idx]
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>"]:
            caption_words.append(word)

    caption = " ".join(caption_words)
    return caption, image
