import time

from model.bag_of_words import BagOfWordsLanguageModel
from training.train import TrainingConfig, launch_training, run_inference

save_path = f'training/checkpoints/bow_{time.strftime("%Y%m%d_%H%M%S")}.pt'
config = TrainingConfig(
    # Model details
    model_type=BagOfWordsLanguageModel,
    iterations=5000,
    seed=42,
    save_path=save_path,
    vocab_file='data/tokenizer/all_chars.json',

    # Dataset details
    data_file='data/corpus/shakespeare.txt',
    max_seq_len=32,
    batch_size=32,
    split=0.9,

    # Logging
    project='bow',
    group='shakespeare',
)
model = launch_training(config)
print(
    run_inference(
        config=config,
        model_ckpt=save_path,
        input_str=' ',
        tokens_to_generate=400,
    ),
)
