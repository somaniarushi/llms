from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """
    A bigram language model that predicts the next token
    given the current token. It's arch is a single lookup table
    that maps from the current token to the next token.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        # each token directly reads off the logits for the next
        # token from a lookup table
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given a batch of sequences of tokens, idx,
        predict the next token in the sequence.
        If targets is provided, compute the cross-entropy loss between the predicted
        next token and the actual next token.
        Else, return the logits for the next token.
        """
        assert idx.dim() == 2
        # idx and targets are both (batch, seq_len) tensor of integers
        logits = self.embedding(idx)  # (batch, seq_len, vocab_size)
        return logits

    def generate(self, idx: torch.Tensor, max_new_tokens: int = 10) -> torch.Tensor:
        """
        Given a batch of sequences of tokens, idx, generate the next max_new_tokens
        tokens for each sequence in the batch.
        """
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
