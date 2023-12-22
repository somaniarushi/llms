from __future__ import annotations

from pathlib import Path

import torch

from data.dataset.base import BaseDataset, BaseDatasetProvider, Batch
from data.tokenizer.base import BaseTokenizer


class TensorDataset(BaseDataset):
    def __init__(
        self,
        data: torch.Tensor,
        tokenizer: BaseTokenizer,
        batch_size: int,
        max_seq_len: int,
    ) -> None:
        self.data = data
        print(f'TensorDataset: {self.data.shape}')
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def __getitem__(self, index: int) -> Batch:
        # TODO: Implement the ability to wrap and do epochs

        # Return a batch of data of the shape (batch_size, max_seq_len)
        # The input and target tensors should be offset by one timestep
        # from each other.

        # Get the batch of data at the specified index
        batch = self.data[:, index, :]
        assert batch.shape == (self.batch_size, self.max_seq_len), (
            f'{batch.shape} != {(self.batch_size, self.max_seq_len)}'
        )
        # Get the input and target tensors
        target = batch
        assert target.shape == (self.batch_size, self.max_seq_len), (
            f'{target.shape} != {(self.batch_size, self.max_seq_len)}'
        )
        # For every last token in the batch, drop it
        input_intermediate = batch[:, :-1]
        assert input_intermediate.shape == (self.batch_size, self.max_seq_len - 1), (
            f'{input_intermediate.shape} != {(self.batch_size, self.max_seq_len - 1)}'
        )
        # Add a padding token to the beginning of every sequence
        padding_token = torch.tensor(
            [self.tokenizer.padding_token],
            dtype=torch.long,
        )
        padding_token = padding_token.repeat(self.batch_size, 1)
        assert padding_token.shape == (self.batch_size, 1), (
            f'{padding_token.shape} != {(self.batch_size, 1)}'
        )
        input = torch.cat([padding_token, input_intermediate], dim=1)
        assert input.shape == (self.batch_size, self.max_seq_len), (
            f'{input.shape} != {(self.batch_size, self.max_seq_len)}'
        )
        return Batch(input, target)


class TensorDatasetProvider(BaseDatasetProvider):
    """
    Defines a dataset that can be used for training a model.
    Returns two datasets, one for training and one for validation—each of
    which can be used to get batches of data of shape (batch_size, max_seq_len).
    The input and target tensors will be offset by one timestep.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def reformat_data(
        cls,
        data: torch.Tensor,
        batch_size: int,
        max_seq_len: int,
        tokenizer: BaseTokenizer,
    ) -> TensorDataset:
        """
        Given a tensor of shape (full_len,), return a tensor of shape
        (full_len // batch_size, batch_size, max_seq_len)
        """
        # Truncate data so that it is evenly divisible by batch_size * max_seq_len
        if len(data) % (batch_size * max_seq_len) != 0:
            data = data[: -(len(data) % (batch_size * max_seq_len))]
            print(f'WARN: Truncated data to length {len(data)}')
        # Add a batch dimension
        data = data.unsqueeze(0)
        # Reshape data to be (batch_size, full_len // batch_size, max_seq_len)
        data = data.view(batch_size, -1, max_seq_len)
        return TensorDataset(
            data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )  # c, b, s

    @classmethod
    def split_data(cls, data: torch.Tensor, split: float):
        split_idx = int(len(data) * split)
        return data[:split_idx], data[split_idx:]

    @classmethod
    def load_data(cls, data_file: Path, tokenizer: BaseTokenizer) -> torch.Tensor:
        with open(data_file) as f:
            raw_data = f.read()
        data = torch.tensor(
            tokenizer.encode(raw_data),
            dtype=torch.long,
        )  # (1115394,)
        return data
