from __future__ import annotations

from pathlib import Path

import torch

from data.dataset.base import BaseDataset, BaseDatasetProvider, Batch
from data.tokenizer.base import BaseTokenizer


class TensorDataset(BaseDataset):
    def __getitem__(self, index: int) -> Batch:
        # Return a batch of data of the shape (batch_size, max_seq_len)
        # The input and target tensors should be offset by one timestep
        # from each other.

        # Get the batch of data at the specified index
        batch = self.data[:, index, :]
        # Get the input and target tensors
        input = batch
        target = batch[:, 1:]
        # Since both need to be (batch_size, max_seq_len), we need to
        # add a padding token to the end of the target tensor
        padding_token = self.tokenizer.padding_token
        target = torch.cat(
            [target, torch.tensor([[padding_token]] * len(target))],
            dim=1,
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
            print(f"WARN: Truncated data to length {len(data)}")
        # Add a batch dimension
        data = data.unsqueeze(0)
        # Reshape data to be (batch_size, full_len // batch_size, max_seq_len)
        data = data.view(batch_size, -1, max_seq_len)
        return TensorDataset(data, tokenizer=tokenizer)  # c, b, s

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
