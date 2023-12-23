import json


def get_vocab_size(file):
    with open(file, 'r') as f:
        vocab = json.load(f)
    return len(vocab)
