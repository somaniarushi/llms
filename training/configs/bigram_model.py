# THIS FILE HAS BEEN DEPRECATED
from model.bigram import BigramLanguageModel
from training.train import TrainingConfig, launch_training, run_inference

config = TrainingConfig(
    # Model details
    model=BigramLanguageModel(
        vocab_size=64,
    ),
    iterations=10000,
    seed=42,
    save_path='training/checkpoints/bigram.pt',
    vocab_file='data/tokenizer/all_chars.json',

    # Dataset details
    data_file='data/corpus/shakespeare.txt',
    max_seq_len=8,
    batch_size=32,
    split=0.9,

    # Logging
    project='bigram',
    group='shakespeare',
)
model = launch_training(config)
print(
    run_inference(
        config=config,
        model_ckpt='training/checkpoints/bigram.pt',
        input_str=' ',
        tokens_to_generate=32,
    ),
)
