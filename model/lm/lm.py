from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lm.attention import AttentionBlock
from model.lm.posemb import PositionalEmbedding


class LanguageModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            embedding_dim: int = 128,
            num_heads: int = 8,
            num_layers: int = 6,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(embedding_dim=embedding_dim, seq_len=seq_len)
        self.attention_blocks = nn.Sequential(
            *[
                AttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    seq_len=seq_len,
                    head_size=embedding_dim // num_heads,
                ) for _ in range(num_layers)
            ],
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim, vocab_size)

    def _embed(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, seq_len)
        -> Embedding -> (batch_size, seq_len, embedding_dim)
        """
        token_embeddings = self.token_embedding(idx) # (batch_size, seq_len, embedding_dim)
        pos_embeddings = self.positional_embedding(idx) # (seq_len, embedding_dim)
        return token_embeddings + pos_embeddings # (batch_size, seq_len, embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Tokens -> Embedding -> Attention Blocks -> Layernorm -> FC -> Logits
        """
        # Embedding step
        embeddings = self._embed(idx)
        logits = self.fc1(self.norm(self.attention_blocks(embeddings)))
        return logits

    def generate(self, idx: torch.Tensor, max_new_tokens: Optional[int] = None) -> torch.Tensor:
        max_new_tokens = max_new_tokens or self.seq_len
        assert max_new_tokens <= self.seq_len, f"Can't do more than seq_len tokens due to positional embeddings"
        # idx is (1, seq_len) tensor of integers
        assert (
            idx.dim() == 2 and idx.shape[0] == 1
        ), f'idx should be (1, seq_len) but got {idx.shape}'
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            assert logits.dim() == 3, (
                f'Expected logits to be (1, seq_len, vocab_size) but got {logits.shape}'
            )
            # Only focus on the last step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx
