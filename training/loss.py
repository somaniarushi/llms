import torch
import torch.nn.functional as F


def cross_entropy_between_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    # The shape of logits is (batch_size, seq_len, vocab_size)
    # The shape of targets is (batch_size, seq_len)
    batch_size, seq_len, vocab_size = logits.shape
    # Flatten logits and targets
    logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(batch_size * seq_len)
    # Compute the cross entropy loss
    return F.cross_entropy(logits_flat, targets_flat)
