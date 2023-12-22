from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """
    Save the model to the given path.
    """
    torch.save(model.state_dict(), path)

def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    """
    Load the model from the given path.
    """
    model.load_state_dict(torch.load(path))
    return model
