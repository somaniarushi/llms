import torch

from data.tokenizer.json_tokenizer import BaseJSONTokenizer


def pretokenize_txt_file(
        txt_file: str,
        out_file: str,
        max_seq_len: int,
) -> None:
    """
    Convert a text file into a tensor dataset, with each "row"
    being a max_seq_len sequence of tokens.
    The dataset is pre-shuffled.
    """
    # Read in the file
    with open(txt_file, 'r') as f:
        text = f.read()
    # Tokenize the text
    tokenizer = BaseJSONTokenizer('data/tokenizer/all_chars.json')
    tokens = tokenizer.encode(text)
    # Convert the tokens into a tensor dataset
    data = torch.tensor(tokens).long()
    if len(data) % max_seq_len != 0:
        # Pad the data
        data = torch.cat([
            data,
            torch.zeros(max_seq_len - (len(data) % max_seq_len)).long(),
        ])
    # Change the data into a tensor of shape (num_sequences, max_seq_len)
    data = data.reshape(-1, max_seq_len)
    # Shuffle the sequences
    data = data[torch.randperm(data.shape[0])]
    # Save the data
    torch.save(data, out_file)

if __name__ == '__main__':
    pretokenize_txt_file(
        txt_file='data/corpus/shakespeare.txt',
        out_file='data/corpus/shakespeare.pt',
        max_seq_len=32,
    )
