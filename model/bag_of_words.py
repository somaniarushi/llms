from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# TODO: Loss is completely flattening, there must be something wrong?
class BagOfWordsLanguageModel(torch.nn.Module):
    def __init__(
        self,
            vocab_size: int,
            embedding_dim: int = 128,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: (batch_size, seq_len)
        -> Embedding -> (batch_size, seq_len, embedding_dim)
        -> FC -> (batch_size, seq_len, vocab_size)
        """
        embeddings = self.embedding(idx) # (batch_size, seq_len, embedding_dim)
        logits = self.fc1(embeddings) # (batch_size, seq_len, vocab_size)
        if targets is None:
            return logits, None
        else:
            batch, seqlen, vocab_size = logits.shape
            logits = logits.reshape(batch * seqlen, vocab_size)
            loss = torch.nn.functional.cross_entropy(logits, targets.reshape(batch * seqlen))
            return self.softmax(logits), loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        assert idx.dim() == 2 and idx.shape[0] == 1, f'idx should be (1, seq_len) but got {idx.shape}'
        input = idx # (1, seq_len)
        for _ in range(max_new_tokens):
            probs = self.forward(input)[0][:, -1, :]
            next_input = torch.multinomial(probs, num_samples=1)
            input = torch.cat([input, next_input], dim=1) # (T + 1, seq_len)
        return input
