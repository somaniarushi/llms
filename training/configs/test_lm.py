import time

from model.lm.lm import LanguageModel
from training.inference import run_inference
from training.train import TrainingConfig, launch_training
from training.utils import get_vocab_size

VOCAB_FILE = 'data/tokenizer/all_chars.json'
DATA_FILE = 'data/corpus/shakespeare.txt'

VOCAB_SIZE = get_vocab_size(VOCAB_FILE)
BATCH_SIZE, SEQ_LEN = 16, 256
SPLIT = 0.9
N_EMBED, N_HEAD, N_LAYER = 64, 8, 4
DROPOUT = 0.0

SAVE_PATH = f'training/checkpoints/lm_{time.strftime("%Y%m%d_%H%M%S")}'

training_config = TrainingConfig(
    model=LanguageModel(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        embedding_dim=N_EMBED,
        num_heads=N_HEAD,
        num_layers=N_LAYER,
        dropout=DROPOUT,
    ),
    data_file=DATA_FILE,
    vocab_file=VOCAB_FILE,
    save_path=SAVE_PATH,
    batch_size=BATCH_SIZE,
    max_seq_len=SEQ_LEN,
    split=SPLIT,
    seed=42,
    # Logging
    project='lm',
    group='shakespeare',
    iterations=2000,
)
# launch_training(training_config)

INFERENCE_PATH = 'training/checkpoints/lm_20231222_194639/final.pt'
# Now let's generate some text!
last_token = ' '
for i in range(15):
    value = run_inference(
        training_config,
        INFERENCE_PATH,
        last_token,
        tokens_to_generate=100,
    )
    print(value)
    last_token = value[-1]
