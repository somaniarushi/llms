from main import EvalConfig, run_eval

# MODEL = "llama3-70b"
# LOADER = "llama"

# MODEL = "gemma-7b-it"
# LOADER = "google"

# MODEL = "gpt-3.5-turbo"
# LOADER = "openai"

MODEL = "mistral-7x8b"
LOADER = "mistral"

# MODEL = "claude-3-sonnet"


def get_llama8b_config(k_shot: int) -> EvalConfig:
    return EvalConfig(
        model_loader=LOADER,
        model=MODEL,
        eval_name="drop",
        num_examples=100,
        k_shots=k_shot,
        output_dir="outputs/mistral_large_drop",
    )


for k_shot in [0, 1, 2, 3, 4, 5, 10, 12, 13, 14, 15]:
    run_eval(get_llama8b_config(k_shot))
