from main import EvalConfig, run_eval

MODEL = "llama3-70b"


from pathlib import Path


def get_llama8b_config(k_shot: int, model_loader: str, model: str) -> EvalConfig:
    Path(f"gpqa_outputs/{model_loader}/{model}").mkdir(parents=True, exist_ok=True)
    return EvalConfig(
        # model_loader="llama",
        model_loader=model_loader,
        # model=MODEL,
        model=model,
        eval_name="gpqa",
        num_examples=10,
        k_shots=k_shot,
        output_dir=f"gpqa_outputs/{model_loader}/{model}",
    )


for k_shot in [0, 1, 2, 3, 4, 5, 10, 20]:
    # run_eval(get_llama8b_config(k_shot, "mistral", "mistral-7b"))
    # run_eval(get_llama8b_config(k_shot, "llama", "llama3-70b"))
    # run_eval(get_llama8b_config(k_shot, "llama", "llama3-8b"))
    # run_eval(get_llama8b_config(k_shot, "google", "gemma-7b-it"))
    run_eval(get_llama8b_config(k_shot, "openai", "gpt-4o"))
    # run_eval(get_llama8b_config(k_shot, "mistral", "mistral-8x7b"))
