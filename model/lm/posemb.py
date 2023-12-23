import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(seq_len, embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        seq_len = idx.shape[1] # idx is (batch_size, seq_len)
        return self.embedding(torch.arange(idx.shape[1]))
