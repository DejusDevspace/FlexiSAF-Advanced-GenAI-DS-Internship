import torch


class CaptionCollate:
    """
    Custom collate function to pad captions to same length in a batch.
    """

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Pad captions in batch to same length.
        Args:
            batch: list of (image, caption) tuples
        Returns:
            images: stacked image tensor (batch_size, 3, 224, 224)
            targets: padded caption tensor (batch_size, max_length)
        """
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.pad_idx
        )

        return images, targets
