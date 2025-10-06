import torch.nn as nn
from .encoder import EncoderCNN
from .decoder import DecoderRNN


class ImageCaptioningModel(nn.Module):
    """
    Complete Image Captioning model combining encoder and decoder.
    """

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers=1
    ):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        """Training forward pass."""
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, max_length=20):
        """Generate caption for a single image."""
        features = self.encoder(image)
        # caption = self.decoder.sample(features, max_length)
        caption = self.decoder.beam_search(features, beam_size=8)
        return caption
