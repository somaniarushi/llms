
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.lm.ffn import FeedForward


class AttentionHead(nn.Module):
    """
    A singular self-attention head.
    """
    def __init__(
        self,
            head_size: int,
            embedding_dim: int,
            seq_len: int,
            dropout: float,
    ):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.tril = torch.tril(torch.ones(seq_len, seq_len))

    def _attention_scores(
            self,
            key: torch.Tensor,
            query: torch.Tensor,
            embedding_dim: int,
            seq_len: int,
    ) -> torch.Tensor:
        """
        The formula for attention calculation is
        SOFTMAX(MASK((QK^T) / sqrt(d)))
        """
        wei = query @ key.transpose(-2,-1) * embedding_dim ** -0.5
        masked_wei = wei.masked_fill(self.tril[:seq_len,:seq_len] == 0, float('-inf'))
        return F.softmax(masked_wei, dim=-1) # (batch_size, seq_len, seq_len)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embedding_dim = idx.shape
        key, query, value = self.key(idx), self.query(idx), self.value(idx) # (batch_size, seq_len, head_size)
        # Compute attention scores
        attention_scores = self._attention_scores(key, query, embedding_dim, seq_len) # (batch_size, seq_len, seq_len)
        # Apply dropout
        attention_scores = self.dropout(attention_scores)
        # Return weighted sum of values
        return attention_scores @ value

class MultiHeadAttention(nn.Module):
    """
    Parallel self-attention heads.
    """
    def __init__(
            self,
            num_heads: int,
            head_size: int,
            embedding_dim: int,
            seq_len: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(head_size, embedding_dim, seq_len, dropout)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(idx) for head in self.heads], dim=-1) # (batch_size, seq_len, embedding_dim)
        out = self.projection(out)
        return self.dropout(out)

class AttentionBlock(nn.Module):
    """
    A single attention block, which consists of
    one multi-head attention layer and one feed-forward layer,
    with a couple layer norms in the middle.
    [LayerNorm -> Attention Block -> LayerNorm -> FFWD -> Out]
    """
    def __init__(
            self,
            num_heads: int,
            head_size: int,
            embedding_dim: int,
            seq_len: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, head_size, embedding_dim, seq_len, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffwd = FeedForward(embedding_dim, embedding_dim, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        out = self.attention(self.norm1(idx))
        out = self.norm2(out + self.ffwd(out))
        return out
