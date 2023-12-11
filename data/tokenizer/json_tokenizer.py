from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from data.tokenizer.base import BaseTokenizer


class BaseJSONTokenizer(BaseTokenizer):
    """
    Base Tokenizer class that assumes the the vocab file is a JSON file,
    where the keys are the tokens and the values are the indices.
    """

    def load_vocab(self, vocab_file: Path) -> dict[str, int]:
        with open(vocab_file) as f:
            vocab = json.load(f)
        return vocab

    def encode(self, text: str) -> list[int]:
        # Given string, make list of single characters
        chars = list(text)
        assert all(
            len(char) == 1 for char in chars
        ), "Input text must be a string of single characters."
        return [self.stoi(token) for token in chars]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.itos(token) for token in tokens])

    def stoi(self, token: str) -> int:
        if token not in self.vocab:
            raise ValueError(f"Token {token} not in vocab.")
        return self.vocab[token]

    @lru_cache(maxsize=1000)
    def itos(self, index: int) -> str:
        for token, idx in self.vocab.items():
            if idx == index:
                return token
        raise ValueError(f"Index {index} not in vocab.")
