from main import EvalConfig, run_eval


def get_sonnet_config(k_shot: int) -> EvalConfig:
    return EvalConfig(
        model_loader="anthropic",
        model="claude-3-opus",
        eval_name="mmlu",
        num_examples=1000,
        k_shots=k_shot,
        output_dir="output",
    )


run_eval(get_sonnet_config(0))
