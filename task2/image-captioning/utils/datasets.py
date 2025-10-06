import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .vocabulary import Vocabulary
from PIL import Image


class FlickrDataset(Dataset):
    """
    Dataset class for Flickr dataset.
    """

    def __init__(
        self,
        root_dir: str,
        captions_file: str,
        transform=None,
        freq_threshold: int = 3
    ):
        self.root_dir = root_dir
        self.df = self.load_captions(captions_file)
        self.transform = transform

        # Get unique image filenames
        self.imgs = self.df["image"].unique()

        # Build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df["caption"].tolist())

    def load_captions(self, captions_file):
        """Load captions from file."""
        df = pd.read_csv(captions_file)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get image-caption pair.
        Returns:
            image: transformed image tensor
            caption: numericalized caption with <start> and <end> tokens
        """
        caption = self.df.iloc[idx]["caption"]
        img_name = self.df.iloc[idx]["image"]
        img_path = os.path.join(self.root_dir, "Images", img_name)

        # Load and transform image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Convert caption to numerical indices
        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized_caption)
