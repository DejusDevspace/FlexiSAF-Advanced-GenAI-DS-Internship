from collections import Counter
import nltk
# nltk.download('punkt')
import string


class Vocabulary:
    """
    Vocabulary class to convert between words and indices.
    Builds vocabulary from training captions.
    """

    def __init__(self, freq_threshold: int = 3):
        # Special tokens
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list: list):
        """
        Build vocabulary from list of captions.
        Only includes words that appear at least 'freq_threshold' times.
        """
        frequencies = Counter()

        # Start after special tokens
        idx = 4

        # Count word frequencies
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                # Add word to vocabulary if threshold is met
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def tokenize(self, text: str):
        """Convert text to lowercase tokens and remove punctuation."""
        tokens = nltk.tokenize.word_tokenize(text.lower())
        tokens = [tok for tok in tokens if tok not in string.punctuation]
        return tokens

    def numericalize(self, text: str):
        """
        Convert text to list of indices.
        Unknown words are mapped to <unk> token.
        """
        tokenized = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized]
