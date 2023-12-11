from typing import NamedTuple

import torch
import torch.nn as nn

from data.dataset.base import BaseDataset
from data.dataset.tensor_loader import TensorDatasetProvider
from data.tokenizer.base import BaseTokenizer
from data.tokenizer.json_tokenizer import BaseJSONTokenizer
from model.bigram import BigramLanguageModel
from training.checkpointing import save_checkpoint


class TrainingConfig(NamedTuple):
    iterations: int
    max_seq_len: int
    batch_size: int
    split: float
    vocab_file: str
    data_file: str
    save_path: str
    model_cls: nn.Module
    seed: int = 42


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
    model = config.model_cls(vocab_size=len(tokenizer))

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
    config = TrainingConfig(
        iterations=10000,
        max_seq_len=8,
        batch_size=4,
        split=0.9,
        vocab_file='data/tokenizer/all_chars.json',
        data_file='data/corpus/shakespeare.txt',
        save_path='training/checkpoints/bigram.pt',
        model_cls=BigramLanguageModel,
    )
    model = launch_training(config)
    print(
        'the', generate(model, 'the', BaseJSONTokenizer(vocab_file=config.vocab_file)),
    )
