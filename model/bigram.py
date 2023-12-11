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
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Given a batch of sequences of tokens, idx,
        ßpredict the next token in the sequence.
        If targets is provided, compute the cross-entropy loss between the predicted
        next token and the actual next token.
        Else, return the logits for the next token.
        """
        assert idx.dim() == 2
        # idx and targets are both (batch, seq_len) tensor of integers
        logits = self.embedding(idx)  # (batch, seq_len, vocab_size)
        b, s, c = logits.shape
        logits = logits.view(b * s, c)  # Flatten the batch and sequence dimensions

        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits, targets.view(b * s))  # Flatten the targets
            return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Given a batch of sequences of tokens, idx, generate the next max_new_tokens
        tokens for each sequence in the batch.
        """
        # idx is (1, seq_len) tensor of integers
        assert idx.dim() == 2 and idx.shape[0] == 1, \
            f'idx should be (1, seq_len) but got {idx.shape}'
        logits, _ = self.forward(idx)
        probs = F.softmax(logits, dim=-1)
        new_tokens = torch.multinomial(probs, max_new_tokens)
        return new_tokens
