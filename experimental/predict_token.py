from __future__ import annotations

from data.dataset.tensor_loader import TensorDatasetProvider
from data.tokenizer.json_tokenizer import BaseJSONTokenizer

max_seq_len = 50
tokenizer = BaseJSONTokenizer('data/tokenizer/all_chars.json')
data = TensorDatasetProvider.get_train_and_val_data(
    data_file='data/corpus/shakespeare.txt',
    tokenizer=tokenizer,
    max_seq_len=max_seq_len,
    batch_size=4,
    split=0.9,
)
train = data.train
valid = data.val
print(f'Train: {train.data.shape} | Valid: {valid.data.shape}')
def print_index(index):
    batch = train[index]
    first_input = batch.input[0]
    first_target = batch.target[0]
    print('In token space:')
    for i in range(max_seq_len):
        print(
            f'When input is {first_input[:i].tolist()}, target is {first_target[i]}',
        )
    print('In character space:')
    for i in range(max_seq_len):
        print(
            f'When input is {tokenizer.decode(first_input[:i])} '
            f'target is {tokenizer.decode([first_target[i]])}',
        )

    print(f'Input to model:\n{input}')

# while True:
#     index = int(input('Enter index: '))
#     print_index(index)

for i in range(1000):
    batch = train[i]
    input = batch.input
    for j in range(4):
        print(f'{tokenizer.decode(input[j].tolist())}')
    print()
