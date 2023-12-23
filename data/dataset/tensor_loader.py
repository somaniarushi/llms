from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Tuple

import torch

from data.tokenizer.base import BaseTokenizer


class Batch(NamedTuple):
    input: torch.Tensor
    target: torch.Tensor
class TrainValidationData(NamedTuple):
    train: TensorDataset
    val: TensorDataset
    split: float
    max_seq_len: int
    tokenizer: BaseTokenizer


class TensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        tokenizer: BaseTokenizer,
        batch_size: int,
        max_seq_len: int,
    ) -> None:
        self.data = data # A tensor of shape (some_size, )
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def __getitem__(self, index: int) -> Batch:
        # Each self.data[idx] is a sequence of max_seq_len tokens
        # We want to create a batch of size self.batch_size
        batch = [self.data[idx] for idx in range(index, index + self.batch_size)]
        # input is batch but the last token is removed and a padding token is added at the beginning
        input = torch.stack([seq[:-1] for seq in batch])
        input = torch.cat(
            [
                torch.zeros((self.batch_size, 1)).long(),
                input,
            ], dim=1,
        )
        # target is just batch
        target = torch.stack(batch)
        return Batch(input=input, target=target)

    def __len__(self) -> int:
        """
        How many batches of max_seq_len tokens can we generate from this dataset?
        """
        return self.data.shape[0] // self.batch_size

class TensorDatasetProvider:
    """
    Defines a dataset that can be used for training a model.
    Returns two datasets, one for training and one for validation—each of
    which can be used to get batches of data of shape (batch_size, max_seq_len).
    The input and target tensors will be offset by one timestep.
    """
    @staticmethod
    def get_train_and_val_data(
        data_file: str,
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        batch_size: int,
        split: int,
    ) -> TrainValidationData:
        data: torch.Tensor = torch.load(data_file)
        assert data.shape[1] == max_seq_len, (
            f'Expected max_seq_len to be {max_seq_len} but got {data.shape[1]}'
        )
        train_size = int(data.shape[0] * split)
        train_data, val_data = data[:train_size], data[train_size:]
        return TrainValidationData(
            train=TensorDataset(
                data=torch.Tensor(train_data).long(),
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            ),
            val=TensorDataset(
                data=torch.Tensor(val_data).long(),
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            ),
            split=split,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )
