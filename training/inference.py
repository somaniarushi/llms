from typing import Optional

import torch
import torch.nn as nn

from data.tokenizer.base import BaseTokenizer
from data.tokenizer.json_tokenizer import BaseJSONTokenizer
from training.checkpointing import load_checkpoint
from training.train import TrainingConfig


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

def run_inference(
    config: TrainingConfig,
    model_ckpt: str,
    input_str: str,
    tokens_to_generate: Optional[int] = None,
) -> str:
    torch.manual_seed(config.seed)
    # load the tokenizer
    tokenizer = BaseJSONTokenizer(vocab_file='data/tokenizer/all_chars.json')

    # load the model
    model = config.model
    model = load_checkpoint(model, model_ckpt)

    # generate some text
    tokens_to_generate = tokens_to_generate or config.max_seq_len
    generated_text = generate(model, input_str, tokenizer, tokens_to_generate)
    return generated_text
