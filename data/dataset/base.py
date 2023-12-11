from pathlib import Path
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from data.tokenizer.base import BaseTokenizer
from typing import NamedTuple, Any, Tuple
import torch
from enum import Enum
from functools import partial


class Batch(NamedTuple):
    input: torch.Tensor
    target: torch.Tensor


class Mode(Enum):
    TRAIN = "train"
    VAL = "val"


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        data: torch.Tensor,
        tokenizer: BaseTokenizer,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer

    @abstractmethod
    def __getitem__(self, index) -> Any:
        raise NotImplementedError


class TrainValidationData(NamedTuple):
    train: BaseDataset
    val: BaseDataset
    split: float
    max_seq_len: int
    tokenizer: BaseTokenizer


class BaseDatasetProvider(ABC):
    """
    Defines a dataset that can be used for training a model.
    When normally instantiated, the getitem function of the dataset will return a batch of data, each
    of shape (batch_size, max_seq_len). The input and target tensors will be offset by one timestep.

    The dataset can be put "in validation mode" by using the validation_mode context manager. This will
    cause the getitem function to return a batch of data, each of shape (batch_size, max_seq_len) from the
    validation cut of the data.
    Usage:
    dataset.validation()

    An error will be raised if validation code is used when split == 1.0.

    """

    @classmethod
    def get_train_and_val_data(
        cls,
        data_file: str,
        tokenizer: BaseTokenizer,
        max_seq_len: int = 8,
        batch_size: int = 4,
        split: float = 0.9,
    ) -> TrainValidationData:
        data = cls.load_data(data_file, tokenizer)
        raw_train_data, raw_val_data = cls.split_data(data, split)
        data_formatter = partial(
            cls.reformat_data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )
        train_data = data_formatter(raw_train_data)
        val_data = data_formatter(raw_val_data)
        return TrainValidationData(
            train=train_data,
            val=val_data,
            split=split,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )

    @classmethod
    @abstractmethod
    def reformat_data(
        cls,
        data: torch.Tensor,
        batch_size: int,
        max_seq_len: int,
        tokenizer: BaseTokenizer,
    ) -> BaseDataset:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def split_data(cls, data: torch.Tensor, split: float):
        raise NotImplementedError

    @classmethod
    def load_data(cls, data_file: Path, tokenizer: BaseTokenizer) -> torch.Tensor:
        raise NotImplementedError
