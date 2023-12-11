import argparse
from typing import NamedTuple

import torch
import torch.nn as nn
import yaml

from data.dataset.base import BaseDataset
from data.dataset.tensor_loader import TensorDatasetProvider
from data.tokenizer.base import BaseTokenizer
from data.tokenizer.json_tokenizer import BaseJSONTokenizer
from model.bigram import BigramLanguageModel
from training.checkpointing import save_checkpoint

STR2MODEL = {
    'bigram': BigramLanguageModel,
}


def load_yaml(config_file: str) -> dict:
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class TrainingConfig(NamedTuple):
    iterations: int
    max_seq_len: int
    batch_size: int
    split: float
    vocab_file: str
    data_file: str
    save_path: str
    model_str: str
    seed: int = 42

    def __post_init__(self) -> None:
        assert self.model_str in STR2MODEL, f'Invalid model string {self.model_str}'


def train(
    model: nn.Module,
    iterations: int,
    dataset: BaseDataset,
    tokenizer: BaseTokenizer,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Train the model for the specified number of iterations.
    """
    # set the model to train mode
    model.train()
    for i in range(iterations):
        batch = dataset[i]

        # get the input and target tokens
        input_tokens = batch.input
        target_tokens = batch.target

        # zero out the gradients
        optimizer.zero_grad()

        # forward pass
        logits, loss = model(input_tokens, target_tokens)

        # backward pass
        loss.backward()
        optimizer.step()

        # print the loss every 100 iterations
        if i % 100 == 0:
            print(f'Iteration {i} loss: {loss.item()}')


def generate(
    model: nn.Module,
    input_str: str,
    tokenizer: BaseTokenizer,
):
    # set the model to eval mode
    model.eval()
    # generate some text
    input_tokens = tokenizer.encode(input_str)
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)
    generated_tokens = model.generate(input_tokens, 10)
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
    model_cls = STR2MODEL[config.model_str]
    model = model_cls(vocab_size=len(tokenizer))

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # train the model
    train(
        model=model,
        iterations=config.iterations,
        dataset=dataset.train,
        tokenizer=dataset.tokenizer,
        optimizer=optimizer,
    )

    # Save the model
    save_checkpoint(model, config.save_path)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = TrainingConfig(**load_yaml(args.config))
    model = launch_training(config)
    print(
        'the',
        generate(model, 'the', BaseJSONTokenizer(vocab_file=config.vocab_file)),
    )
