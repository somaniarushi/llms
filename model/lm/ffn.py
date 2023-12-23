import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    A simple feed-forward network.
    """
    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential([
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        ])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.net(idx)
