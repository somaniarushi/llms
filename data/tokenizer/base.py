import json
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path


class BaseTokenizer(ABC):
    def __init__(
        self, vocab_file: str, eot: str = "|ENDOFTEXT|", padding: str = "|PADDING|"
    ) -> None:
        self.vocab_file = Path(vocab_file)
        self.vocab = self.load_vocab(vocab_file)
        self.eot = eot
        self.padding = padding

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

    @property
    def eot_token(self) -> int:
        return self.stoi(self.eot)

    @property
    def padding_token(self) -> int:
        return self.stoi(self.padding)
