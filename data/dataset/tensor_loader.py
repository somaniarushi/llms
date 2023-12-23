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
        # Wrap around if we get to the end of the dataset
        # IX is batch_size indices into the data
        ix = torch.randint(0, len(self.data) - self.max_seq_len, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.max_seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.max_seq_len+1] for i in ix])
        return Batch(input=x, target=y)
        # index = index if index < len(self) else index % len(self)
        # # Get the start and end indices for the batch
        # start_idx = index * self.batch_size * self.max_seq_len
        # end_idx = start_idx + self.batch_size * self.max_seq_len
        # # Get the batch
        # batch = self.data[start_idx:end_idx]
        # # For the input, remove the last token and add a padding token to the front
        # input = torch.cat(
        #     [
        #         torch.tensor([self.tokenizer.padding_idx]).repeat(self.batch_size, 1),
        #         batch[:, :-1],
        #     ],
        #     dim=1,
        # )
        # target = batch
        # return Batch(input=input, target=target)

    def __len__(self) -> int:
        """
        How many batches of max_seq_len tokens can we generate from this dataset?
        """
        return self.data.shape[0] // (self.batch_size * self.max_seq_len)
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
        with open(Path(data_file), 'r') as f:
            data_raw = f.read()
        data = tokenizer.encode(data_raw)
        train_size = int(len(data) * split)
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
