from main import EvalConfig, run_eval

# MODEL = "gemma-7b-it"
MODEL = "gemma-7b-it"
LOADER = "google"


def get_llama8b_config(k_shot: int) -> EvalConfig:
    return EvalConfig(
        model_loader="google",
        model=MODEL,
        eval_name="mmlu",
        num_examples=100,
        k_shots=k_shot,
        output_dir="outputs/gemma",
    )


# for k_shot in [0, 1, 2, 3, 4, 5, 10, 20]:
for k_shot in [30, 50, 100]:
    run_eval(get_llama8b_config(k_shot))
