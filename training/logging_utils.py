import torch

import wandb
from data.tokenizer.base import BaseTokenizer


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
