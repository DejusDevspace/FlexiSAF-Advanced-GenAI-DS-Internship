import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from eval.helpers import load_model, generate_caption
from eval.visualizations import visualize_prediction, create_demo_grid

def run_inference(checkpoint_path, image_path: str | None, multiple=False):
    """Run inference on a single image using a trained model."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, vocab = load_model(checkpoint_path, device)
    print("Model loaded successfully!")

    if multiple:
        # For multiple images, use create_demo_grid
        image_paths = [
            "../dataset/Images/95728664_06c43b90f1.jpg",
            "../dataset/Images/124881487_36e668145d.jpg",
            "../dataset/Images/101654506_8eb26cfb60.jpg",
            "../dataset/Images/112178718_87270d9b4d.jpg"
        ]
        create_demo_grid(image_paths, model, vocab, device)
        return
    # Generate caption from image and save
    visualize_prediction(image_path, model, vocab, device)


if __name__ == "__main__":
    # Run demo inference
    checkpoint_path = "../artifacts/models/image_captioning_model4.pth"
    image_path = "../dataset/Images/112178718_87270d9b4d.jpg"
    run_inference(checkpoint_path, multiple=True, image_path=None)
