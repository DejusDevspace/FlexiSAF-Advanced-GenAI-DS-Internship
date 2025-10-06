import torch
from datasets import FlickrDataset
from captions import CaptionCollate

def get_data_loader(
    root_dir: str,
    captions_file: str,
    transform,
    batch_size=32,
    num_workers=4,
    shuffle=True
):
    """Helper to create DataLoader for training/validation."""
    dataset = FlickrDataset(root_dir, captions_file, transform)
    pad_idx = dataset.vocab.stoi["<pad>"]

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=CaptionCollate(pad_idx=pad_idx)
    )

    return loader, dataset
