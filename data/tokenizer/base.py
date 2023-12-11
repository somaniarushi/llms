import json
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path

class BaseTokenizer(ABC):
    def __init__(self, vocab_file: str) -> None:
        self.vocab_file = Path(vocab_file)
        self.vocab = self.load_vocab(vocab_file)

    @abstractmethod
    def load_vocab(self, vocab_file: Path) -> Dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def stoi(self, token: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def itos(self, index: int) -> str:
        raise NotImplementedError

class BaseJSONTokenizer(BaseTokenizer, ABC):
    """
    Base Tokenizer class that assumes the the vocab file is a JSON file,
    where the keys are the tokens and the values are the indices.
    """
    def load_vocab(self, vocab_file: Path) -> Dict[str, int]:
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        return vocab

    def encode(self, text: str) -> List[int]:
        return [self.stoi(token) for token in text.split()]

    def decode(self, tokens: List[int]) -> str:
        return " ".join([self.itos(token) for token in tokens])

    def stoi(self, token: str) -> int:
        return self.vocab[token]

    @lru_cache(maxsize=1000)
    def itos(self, index: int) -> str:
        for token, idx in self.vocab.items():
            if idx == index:
                return token
        raise ValueError(f"Index {index} not in vocab.")