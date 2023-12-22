from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTokenizer(ABC):
    def __init__(
        self,
        vocab_file: str,
        eot: str = '|ENDOFTEXT|',
        padding: str = ' ', # TODO: Change to |PAD| token on which we don't take loss
    ) -> None:
        self.vocab_file = Path(vocab_file)
        self.vocab = self.load_vocab(self.vocab_file)
        self.eot = eot
        self.padding = padding

    def __len__(self) -> int:
        return len(self.vocab)

    @abstractmethod
    def load_vocab(self, vocab_file: Path) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
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
