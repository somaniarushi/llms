from __future__ import annotations

import torch

from data.dataset.base import Batch
from data.dataset.tensor_loader import TensorDataset, TensorDatasetProvider
from data.tokenizer.json_tokenizer import BaseJSONTokenizer

SPLIT_ERROR_MARGIN = 1e-4


class TestTensorDataset:
    def test_train_val(self) -> None:
        dataset = TensorDatasetProvider.get_train_and_val_data(
            data_file='data/corpus/shakespeare.txt',
            tokenizer=BaseJSONTokenizer('data/tokenizer/all_chars.json'),
            max_seq_len=8,
            batch_size=4,
            split=0.9,
        )
        assert isinstance(
            dataset.train,
            TensorDataset,
        ), 'Expected train dataset to be a TensorDataset'
        assert isinstance(
            dataset.val,
            TensorDataset,
        ), 'Expected val dataset to be a TensorDataset'

        assert dataset.train.data.shape == (4, 31370, 8)
        assert dataset.val.data.shape == (4, 3485, 8)

        split_ratio = dataset.train.data.shape[1] / (
            dataset.val.data.shape[1] + dataset.train.data.shape[1]
        )
        assert abs(split_ratio - dataset.split) < SPLIT_ERROR_MARGIN

        assert len(dataset.train.data[0][0]) == dataset.max_seq_len
        assert len(dataset.val.data[0][0]) == dataset.max_seq_len

        # First batch, first sequence
        assert torch.allclose(
            dataset.train.data[0][0],
            torch.tensor([18, 47, 56, 57, 58, 1, 15, 47]),
        )
        assert dataset.tokenizer.decode(dataset.train.data[0][0]) == 'First Ci'

    def test_getitem(self) -> None:
        dataset = TensorDatasetProvider.get_train_and_val_data(
            data_file='data/corpus/shakespeare.txt',
            tokenizer=BaseJSONTokenizer('data/tokenizer/all_chars.json'),
            max_seq_len=8,
            batch_size=4,
            split=0.9,
        )
        batch = dataset.train[0]
        assert isinstance(batch, Batch)
        assert batch.input.shape == (4, 8)
        assert batch.target.shape == (4, 8)
        assert torch.allclose(
            batch.input[0],
            torch.tensor([18, 47, 56, 57, 58, 1, 15, 47]),
        )
        assert torch.allclose(
            batch.target[0],
            torch.tensor([47, 56, 57, 58, 1, 15, 47, 1]),
        )
        assert dataset.tokenizer.decode(batch.input[0]) == 'First Ci'
        assert dataset.tokenizer.decode(batch.target[0]) == 'irst Ci|PADDING|'

    def test_full_batch(self) -> None:
        dataset = TensorDatasetProvider.get_train_and_val_data(
            data_file='data/corpus/shakespeare.txt',
            tokenizer=BaseJSONTokenizer('data/tokenizer/all_chars.json'),
            max_seq_len=8,
            batch_size=4,
            split=0.9,
        )
        batch = dataset.train[755]
        decoded_inputs = [dataset.tokenizer.decode(input) for input in batch.input]
        decoded_targets = [dataset.tokenizer.decode(target) for target in batch.target]
        assert decoded_inputs == [' proceed', 'tizens a', 'a husban', 'S:\nI am ']
        assert decoded_targets == ['proceed ', 'izens a ', ' husban ', ':\nI am  ']
