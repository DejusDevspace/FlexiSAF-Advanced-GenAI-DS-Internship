import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    """
    The decoder generates captions word-by-word using LSTM.
    It takes image features and previously generated words as input.
    """
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers=1
    ):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Convert word indices to vectors
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Define LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # Linear layer to map LSTM output to vocabulary
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
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        # Pass through LSTM
        hidden, _ = self.lstm(embeddings)

        # Generate predictions for each word
        outputs = self.linear(hidden)
        return outputs

    def sample(self, features, max_length: int = 20):
        """
        Generate captions using greedy search (for inference).
        Args:
            features: image features (batch_size, embed_size)
            max_length: maximum caption length
        Returns:
            captions: list of word indices
        """
        captions = []
        inputs = features.unsqueeze(1)
        states = None  # Initial hidden state

        for i in range(max_length):
            # Forward pass through LSTM
            hidden, states = self.lstm(inputs, states)
            outputs = self.linear(hidden.squeeze(1))

            # Get word with the highest probability
            _, predicted = outputs.max(1)
            captions.append(predicted.item())

            # Use predicted word as input for next step
            inputs = self.embed(predicted).unsqueeze(1)

            # Stop if <end> token is generated
            if predicted.item() == 2:  #  <end> token
                break

        return captions

    def beam_search(self, features, beam_size=3, max_length=20):
        """
        Generate captions using beam search.
        Args:
            features: image features (1, embed_size)
            beam_size: number of beams
            max_length: maximum caption length
        Returns:
            best_caption: list of word indices
        """
        device = features.device

        # Start token index
        start_token = 1
        end_token = 2

        # Initialize beam: (sequence, hidden_state, log_prob)
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        states = None
        beams = [(torch.tensor([start_token], device=device), states, 0.0)]  # log prob = 0

        for _ in range(max_length):
            candidates = []
            for seq, states, score in beams:
                if seq[-1].item() == end_token:
                    # Already ended, keep as is
                    candidates.append((seq, states, score))
                    continue

                # Pass last word through embedding + LSTM
                if seq.size(0) == 1:  # first step: input = image features
                    inputs = features.unsqueeze(1)
                else:
                    inputs = self.embed(seq[-1]).unsqueeze(0).unsqueeze(1)  # (1,1,embed_size)

                hidden, states = self.lstm(inputs, states)
                outputs = self.linear(hidden.squeeze(1))  # (1, vocab_size)
                log_probs = F.log_softmax(outputs, dim=1)

                # Get top beam_size candidates
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=1)

                for k in range(beam_size):
                    next_seq = torch.cat([seq, topk_ids[0, k].unsqueeze(0)])
                    candidates.append((next_seq, states, score + topk_log_probs[0, k].item()))

            # Select top beam_size sequences
            beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_size]

        # Pick the best sequence (highest log prob)
        best_seq = beams[0][0].tolist()
        return best_seq

