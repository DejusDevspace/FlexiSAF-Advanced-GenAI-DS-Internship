import torch
import torch.nn as nn
import torchvision.models as models


class DecoderRNN(nn.Module):
    """
    Decoder generates captions word-by-word using LSTM.
    Takes image features and previously generated words as input.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Word embedding layer converts word indices to vectors
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM generates captions based on image features and previous words
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # Linear layer maps LSTM output to vocabulary
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Training forward pass.
        Args:
            features: image features from encoder (batch_size, embed_size)
            captions: target captions (batch_size, caption_length)
        Returns:
            outputs: predicted word scores (batch_size, caption_length, vocab_size)
        """
        # Embed captions (exclude last word for teacher forcing)
        embeddings = self.embed(captions[:, :-1])

        # Concatenate image features with word embeddings
        # features.unsqueeze(1) adds time dimension: (batch, 1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        # Pass through LSTM
        hiddens, _ = self.lstm(embeddings)

        # Generate predictions for each word
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, max_length=20):
        """
        Generate captions using greedy search (for inference).
        Args:
            features: image features (batch_size, embed_size)
            max_length: maximum caption length
        Returns:
            captions: list of word indices
        """
        captions = []
        inputs = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        states = None  # Initial hidden state

        for i in range(max_length):
            # Forward pass through LSTM
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # Get word with highest probability
            _, predicted = outputs.max(1)
            captions.append(predicted.item())

            # Use predicted word as input for next step
            inputs = self.embed(predicted).unsqueeze(1)

            # Stop if <end> token is generated
            if predicted.item() == 1:  # Assuming 1 is <end> token
                break

        return captions
