from data.dataset.tensor_loader import TensorDatasetProvider, TensorDataset
from data.tokenizer.json_tokenizer import BaseJSONTokenizer

max_seq_len = 8
tokenizer = BaseJSONTokenizer('data/tokenizer/all_chars.json')
data = TensorDatasetProvider.get_train_and_val_data(
    data_file='data/corpus/shakespeare.txt',
    tokenizer=tokenizer,
    max_seq_len=max_seq_len,
    batch_size=4,
    split=0.9,
)
train = data.train
input, target = train[0]
assert input.shape == (4, 8)
assert target.shape == (4, 8)
first_input = input[0]
first_target = target[0]
assert len(first_input) == len(first_target) == 8
print("In token space:")
for i in range(max_seq_len):
    print(f"When input is {first_input[:i].tolist()}, target is {first_target[i]}")
print("In character space:")
for i in range(max_seq_len):
    print(f"When input is {tokenizer.decode(first_input[:i])}, target is {tokenizer.decode([first_target[i]])}")