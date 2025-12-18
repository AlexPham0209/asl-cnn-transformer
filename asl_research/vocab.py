import json
import os

import torch


class TextVocabulary:
    def __init__(self, words: str):
        self.words = self.words

        # Create dictionaries to convert string tokens into their ids and vice versa
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.words)}

        assert "<pad>" in self.word_to_idx, "<PAD> token doesn't exist in text vocab"
        assert "<eos>" in self.word_to_idx, "<EOS> token doesn't exist in text vocab"
        assert "<sos>" in self.word_to_idx, "<SOS> token doesn't exist in text vocab"

        self.sos_token = self.word_to_idx["<sos>"]
        self.eos_token = self.word_to_idx["<eos>"]
        self.pad_token = self.word_to_idx["<pad>"]

    def tokenize(self, sentence: str):
        return torch.tensor(
            [self.word_to_idx["<sos>"]]
            + [self.word_to_idx[word] for word in sentence.split()]
            + [self.word_to_idx["<eos>"]]
        )

    def tokenize_batch(self, sentences: list):
        return torch.stack([self.tokenize(sentence) for sentence in sentences], dim=0)

    def decode(self, sentence: list):
        sentence = list(filter(self.remove_special_tokens, sentence))
        return " ".join([self.idx_to_word[token] for token in sentence])

    def decode_batch(self, sentences: list):
        return [self.decode(sentence) for sentence in sentences]

    def remove_special_tokens(self, token: int):
        return token != self.pad_token and token != self.eos_token and token != self.sos_token

    def get_size(self):
        return len(self.word_to_idx)
    

class GlossVocabulary:
    def __init__(self, glosses: list):
        self.glosses = glosses

        # Create dictionaries to convert string tokens into their ids and vice versa
        self.gloss_to_idx = {gloss: i for i, gloss in enumerate(self.glosses)}
        self.idx_to_gloss = {i: gloss for i, gloss in enumerate(self.glosses)}

        assert "<pad>" in self.gloss_to_idx, "<PAD> token doesn't exist in gloss vocab"
        assert "-" in self.gloss_to_idx, "Blank token doesn't exist in gloss vocab"

        self.blank_token = self.gloss_to_idx["-"]
        self.pad_token = self.gloss_to_idx["<pad>"]
    
    def tokenize(self, gloss: str):
        return torch.tensor([self.gloss_to_idx[word] for word in gloss.split()])

    def tokenize_batch(self, glosses: list):
        return torch.stack([self.tokenize(gloss) for gloss in glosses], dim=0)

    def decode(self, gloss: list):
        gloss = filter(self.remove_special_tokens, gloss)
        return " ".join([self.idx_to_gloss[token] for token in gloss])

    def decode_batch(self, glosses: list):
        return [self.decode(gloss) for gloss in glosses]

    def remove_special_tokens(self, token: int):
        return token != self.pad_token and token != self.blank_token

    def get_size(self):
        return len(self.gloss_to_idx)
