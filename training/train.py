import os
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

import wandb
from data.dataset.base import BaseDataset
from data.dataset.tensor_loader import TensorDatasetProvider
from data.tokenizer.json_tokenizer import BaseJSONTokenizer
from training.checkpointing import save_checkpoint
from training.logging_utils import log_data_to_wandb
from training.loss import cross_entropy_between_logits_and_targets


class TrainingConfig(NamedTuple):
    """
    General Args:
        iterations (int): How many iterations to train for; can be multiple epochs or less than one epoch.
        vocab_file (str): Where the vocab is stored.
        save_path (str): Where to save the model.
        model (nn.Module): Root class of the model to use.
        seed (int): Random seed for reproducibility.
    Data Args:
        max_seq_len (int): How many tokens to use per sequence.
        batch_size (int): How many sequences to use per batch.
        split (float): What percentage of the data to use for training vs validation.
        data_file (str): Where the data is stored.
    Logging Args:
        project (str): Name of the wandb project to log to.
        group (str): Name of the wandb group to log to.
        name (str): Name of the wandb run.
    """
    iterations: int
    max_seq_len: int
    batch_size: int
    split: float
    data_file: str
    vocab_file: str
    save_path: str
    model: nn.Module
    project: str
    group: str
    seed: int = 42

def get_validation_loss(
    model: nn.Module,
    validation: BaseDataset,
) -> None:
    """
    Compute the validation loss and print it out.
    """
    model.eval()

    total_loss = 0
    for i in range(len(validation)):
        batch = validation[i]
        input_tokens = batch.input
        target_tokens = batch.target
        logits = model(input_tokens)
        loss = cross_entropy_between_logits_and_targets(logits, target_tokens)
        total_loss += loss.item()

    model.train()
    return total_loss / len(validation)

def train(
    model: nn.Module,
    iterations: int,
    dataset: BaseDataset,
    validation: BaseDataset,
    checkpoint_root_dir: str,
    optimizer: torch.optim.Optimizer,
    eval_iterations: int = 100,
    save_iterations: int = 50,
    loss_log_iterations: int = 10,
    log_iterations: int = 5000,
) -> None:
    """
    Train the model for the specified number of iterations.
    """
    # set the model to train mode
    model.train()
    for idx in range(iterations):
        batch = dataset[idx]

        # get the input and target tokens
        input_tokens, target_tokens = batch.input, batch.target

        if idx % log_iterations == 0:
            log_data_to_wandb(input_tokens, target_tokens, dataset.tokenizer)

        # zero out the gradients
        optimizer.zero_grad()

        # forward pass
        logits = model(input_tokens)
        loss = cross_entropy_between_logits_and_targets(logits, target_tokens)

        # backward pass
        loss.backward()
        optimizer.step()

        # Log the training loss
        if idx % loss_log_iterations == 0:
            wandb.log({'loss': loss.item()}, step=idx)

        # print the loss every 100 iterations
        if idx % eval_iterations == 0:
            val_loss = get_validation_loss(model, validation)
            wandb.log({'val_loss': val_loss}, step=idx)

        # Save checkpoint if necessary
        if idx % save_iterations == 0:
            save_checkpoint(model, Path(checkpoint_root_dir) / f'iter_{idx}.pt')

def launch_training(
    config: TrainingConfig,
) -> nn.Module:
    # Lock in the seed for reproducibility
    torch.manual_seed(config.seed)

    # load the tokenizer
    tokenizer = BaseJSONTokenizer(vocab_file=config.vocab_file)

    # create the dataset
    dataset = TensorDatasetProvider.get_train_and_val_data(
        data_file=config.data_file,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        split=config.split,
    )
    # create the model
    model = config.model

    # Print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize wandb
    wandb.init(
        project=config.project,
        group=config.group,
        name=f"{config.model.__class__.__name__}_{time.strftime('%Y%m%d_%H%M%S')}",
    )

    # If the checkpoint root doesn't exist, create it
    os.makedirs(config.save_path, exist_ok=True)

    # train the model
    train(
        model=model,
        iterations=config.iterations,
        checkpoint_root_dir=config.save_path,
        dataset=dataset.train,
        validation=dataset.val,
        optimizer=optimizer,
    )

    # Save the model
    save_checkpoint(model, Path(config.save_path + '/final.pt'))
    return model
