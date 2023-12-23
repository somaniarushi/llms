import torch
import torch.nn as nn


class LayerNorm1D(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, embedding_dim: int, eps: float = 1e-08) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        mean = idx.mean(dim=1, keepdim=True)
        var = idx.var(dim=1, keepdim=True)
        idx_norm = (idx - mean) / torch.sqrt(var + self.eps)
        return self.gamma * idx_norm + self.beta
