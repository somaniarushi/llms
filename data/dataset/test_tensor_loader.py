from __future__ import annotations

import torch

from data.dataset.tensor_loader import (Batch, TensorDataset,
                                        TensorDatasetProvider)
from data.tokenizer.json_tokenizer import BaseJSONTokenizer

SPLIT_ERROR_MARGIN = 1e-4


class TestTensorDataset:
    def test_train_val(self) -> None:
        dataset = TensorDatasetProvider.get_train_and_val_data(
            data_file='data/corpus/shakespeare_32.pt',
            tokenizer=BaseJSONTokenizer('data/tokenizer/all_chars.json'),
            max_seq_len=32,
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

        assert dataset.train.data.shape == (31370, 32)
        assert dataset.val.data.shape == (3486, 32)

        split_ratio = len(dataset.train) / (len(dataset.train) + len(dataset.val))
        assert abs(split_ratio - dataset.split) < SPLIT_ERROR_MARGIN

        assert len(dataset.train.data[0]) == dataset.max_seq_len
        assert len(dataset.val.data[0]) == dataset.max_seq_len

        # First batch, first sequence
        assert torch.allclose(
            dataset.train.data[0][:8],
            torch.tensor([46, 43, 63, 1, 44, 47, 52, 42]),
        )
        assert (
            dataset.tokenizer.decode(dataset.train.data[0])
            == 'hey find a kind of ease,\nBearing'
        )

    def test_getitem(self) -> None:
        dataset = TensorDatasetProvider.get_train_and_val_data(
            data_file='data/corpus/shakespeare_32.pt',
            tokenizer=BaseJSONTokenizer('data/tokenizer/all_chars.json'),
            max_seq_len=32,
            batch_size=4,
            split=0.9,
        )
        batch = dataset.train[0]
        assert isinstance(batch, Batch)
        assert batch.input.shape == (4, 32)
        assert batch.target.shape == (4, 32)
        assert torch.allclose(
            batch.input[0][:8],
            torch.tensor([0, 46, 43, 63, 1, 44, 47, 52]),
        )
        assert torch.allclose(
            batch.target[0][:8],
            torch.tensor([46, 43, 63, 1, 44, 47, 52, 42]),
        )
        assert (
            dataset.tokenizer.decode(batch.input[0])
            == '\nhey find a kind of ease,\nBearin'
        )
        assert (
            dataset.tokenizer.decode(batch.target[0])
            == 'hey find a kind of ease,\nBearing'
        )

    def test_full_batch(self) -> None:
        dataset = TensorDatasetProvider.get_train_and_val_data(
            data_file='data/corpus/shakespeare_32.pt',
            tokenizer=BaseJSONTokenizer('data/tokenizer/all_chars.json'),
            max_seq_len=32,
            batch_size=4,
            split=0.9,
        )
        batch = dataset.train[755]
        decoded_inputs = [dataset.tokenizer.decode(input) for input in batch.input]
        decoded_targets = [dataset.tokenizer.decode(target) for target in batch.target]
        assert decoded_inputs == [
            '\nCK:\nThis is his tent; and see w',
            '\nim sleeping.\n\nSecond Murderer:\n',
            '\nth, this realm, this England,\nT',
            '\n it forth\nTo seek the empty, va',
        ]
        assert decoded_targets == [
            'CK:\nThis is his tent; and see wh',
            'im sleeping.\n\nSecond Murderer:\nT',
            'th, this realm, this England,\nTh',
            ' it forth\nTo seek the empty, vas',
        ]
