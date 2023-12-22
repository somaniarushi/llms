from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List

import fire


def get_all_characters_as_tokens(
    corpus_file: Path,
) -> List[str]:
    """
    Given a corpus file, which is a .txt file containing the corpus,
    get all the unique characters from the txt file.
    """
    # Read the corpus file
    with open(corpus_file) as f:
        corpus = f.read()
    return sorted(list(set(corpus)))


def _create_tokenzizer_from_txt(
    corpus_file: Path,
    vocab_file: Path,
    sampling_function: Callable[[Path], list[str]],
) -> bool:
    """
    Given a corpus file, which is a .txt
    file containing the corpus, and a sampling function,
    return the vocab file for the corpus.
    """
    # Get all the unique tokens from the corpus
    tokens = sampling_function(corpus_file)
    print(f'Found {len(tokens)} unique tokens.')
    # Create the mapping from tokens to indices
    vocab = {token: idx for idx, token in enumerate(tokens)}
    # Save the vocab file
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    # Return the response
    return True


def create_simple_tokenizer(
    corpus_file_str: str,
    vocab_file_str: str,
    sampling_strategy: str,
) -> bool:
    """
    Given a corpus file, create a vocab file for it using the
    provided sampling strategy.
    Currently only accepts .txt files and all-characters sampling strategy.
    """
    assert (
        sampling_strategy == 'all_characters'
    ), 'Only all_characters sampling strategy is supported.'
    corpus_file = Path(corpus_file_str)
    # Assert that the corpus file exists and is a .txt file
    assert corpus_file.exists()
    assert corpus_file.suffix == '.txt', 'Only .txt corpus files are supported.'

    vocab_file = Path(vocab_file_str)
    # Assert that the vocab file does not exist
    assert not vocab_file.exists()
    # Assert that the vocab file is a .json file
    assert vocab_file.suffix == '.json', 'Only .json vocab files are supported.'

    # Create the vocab file
    response = _create_tokenzizer_from_txt(
        corpus_file,
        vocab_file,
        get_all_characters_as_tokens,
    )
    return response


if __name__ == '__main__':
    fire.Fire(create_simple_tokenizer)
    # Example usage: python experimental/create_simple_tokenizer.py
    # --corpus_file_str data/corpus/shakespeare.txt
    # --vocab_file_str data/tokenizer/all_chars.json
    # --sampling_strategy all_characters
