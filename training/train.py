import time
from typing import NamedTuple

import torch
import torch.nn as nn
import yaml

import wandb
from data.dataset.base import BaseDataset
from data.dataset.tensor_loader import TensorDatasetProvider
from data.tokenizer.base import BaseTokenizer
from data.tokenizer.json_tokenizer import BaseJSONTokenizer
from training.checkpointing import load_checkpoint, save_checkpoint


def load_yaml(config_file: str) -> dict:
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class TrainingConfig(NamedTuple):
    """
    General Args:
        iterations (int): How many iterations to train for; can be multiple epochs or less than one epoch.
        vocab_file (str): Where the vocab is stored.
        save_path (str): Where to save the model.
        model_type (nn.Module): Root class of the model to use.
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
    model_type: nn.Module
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
        _, loss = model(input_tokens, target_tokens)
        total_loss += loss.item()

    model.train()
    return total_loss / len(validation)

def log_data_to_wandb(
    input_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    tokenizer: BaseTokenizer,
) -> None:
        # Detokenize the first input and target tokens
        input_text = '|END OF TASK|'.join([tokenizer.decode(tokens) for tokens in input_tokens])
        target_text = '|END OF TASK|'.join([tokenizer.decode(tokens) for tokens in target_tokens])
        # Log the input and target text into a table with two columns
        table = wandb.Table(columns=['input', 'target'])
        table.add_data(input_text, target_text)
        wandb.log({'input_target': table})

def train(
    model: nn.Module,
    iterations: int,
    dataset: BaseDataset,
    validation: BaseDataset,
    optimizer: torch.optim.Optimizer,
    eval_iterations: int = 100,
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
        _, loss = model(input_tokens, target_tokens)

        # backward pass
        loss.backward()
        optimizer.step()

        # Log the training loss
        if idx % loss_log_iterations == 0:
            wandb.log({'loss': loss.item()})

        # print the loss every 100 iterations
        if idx % eval_iterations == 0:
            val_loss = get_validation_loss(model, validation)
            wandb.log({'val_loss': val_loss})

def generate(
    model: nn.Module,
    input_str: str,
    tokenizer: BaseTokenizer,
    num_tokens: int = 10,
):
    # set the model to eval mode
    model.eval()
    # generate some text
    input_tokens = tokenizer.encode(input_str)
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)
    generated_tokens = model.generate(input_tokens, num_tokens)
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    return generated_text


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
    model_cls = config.model_type
    model = model_cls(vocab_size=len(tokenizer))

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize wandb
    wandb.init(
        project=config.project,
        group=config.group,
        name=f"{config.model_type.__name__}_{time.strftime('%Y%m%d_%H%M%S')}",
    )

    # train the model
    train(
        model=model,
        iterations=config.iterations,
        dataset=dataset.train,
        validation=dataset.val,
        optimizer=optimizer,
    )

    # Save the model
    save_checkpoint(model, config.save_path)
    return model

def run_inference(
    config: TrainingConfig,
    model_ckpt: str,
    input_str: str,
    tokens_to_generate: int = 10,
) -> str:
    torch.manual_seed(config.seed)
    # load the tokenizer
    tokenizer = BaseJSONTokenizer(vocab_file='data/tokenizer/all_chars.json')

    # load the model
    model = config.model_type(vocab_size=len(tokenizer))
    model = load_checkpoint(model, model_ckpt)

    # generate some text
    generated_text = generate(model, input_str, tokenizer, tokens_to_generate)
    return generated_text
