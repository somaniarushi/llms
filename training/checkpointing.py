from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """
    Save the model to the given path.
    """
    torch.save(model.state_dict(), path)
