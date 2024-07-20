from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evals.mapping import EVAL_TASK_MAPPING, run_eval_from_name
from models.mapping import MODEL_LOADING_MAP, get_model_server
from typings import EvalResult


@dataclass(frozen=True)
class EvalConfig:
    """
    Configuration for an evaluation job. To use, specify:
    model_loader: "openai" or "anthropic"
    model: The model to use, based on the model loader
    eval_name: The evaluation task to run
    num_examples: How many datapoints of the eval to run
    k_shots: How many shots to use for the eval
    output_dir: Where to save the results

    Please view MODEL_LOADING_MAP and EVAL_TASK_MAPPING for list of supported models and eval tasks.
    """

    model_loader: str
    model: str
    eval_name: str
    num_examples: int
    k_shots: int
    output_dir: str

    def __post_init__(self) -> None:
        assert (
            self.model_loader in MODEL_LOADING_MAP
        ), f"Unknown {self.model_loader=} | Accepted values: {MODEL_LOADING_MAP.keys()}"
        assert (
            self.eval_name in EVAL_TASK_MAPPING
        ), f"Unknown {self.eval_name=} | Accepted values: {EVAL_TASK_MAPPING.keys()}"

        model_loader_models = MODEL_LOADING_MAP[self.model_loader]
        assert (
            self.model in model_loader_models
        ), f"Unknown {self.model=} | Accepted values: {model_loader_models.keys()}"

        if not Path(self.output_dir).exists():
            print(f"Output directory {self.output_dir} does not exist. Creating it...")
            Path(self.output_dir).mkdir(parents=True)


def run_eval(eval_config: EvalConfig) -> None:
    """
    Central eval runner
    """
    server = get_model_server(eval_config.model_loader, eval_config.model)
    print(f"Using server: {server}")
    eval_result: EvalResult = run_eval_from_name(
        eval_name=eval_config.eval_name,
        server=server,
        num_examples=eval_config.num_examples,
        k_shots=eval_config.k_shots,
    )

    file_slug = f"{eval_config.eval_name}_{eval_config.model}_fs{eval_config.k_shots}_{eval_config.num_examples}"
    # First, save out a summary of the score and metrics
    if eval_result.metrics is not None:
        summary_out_path = Path(eval_config.output_dir) / f"{file_slug}_summary.txt"
        print(f"Saving evaluation summary to {summary_out_path}")
        with open(summary_out_path, "w", encoding="utf-8") as f:
            f.write(f"Score: {eval_result.score}\n")
            for metric, value in eval_result.metrics.items():
                f.write(f"{metric}: {value}\n")

    # Next, save out the conversations
    csv_out_path = Path(eval_config.output_dir) / f"{file_slug}_conversations.csv"
    print(f"Saving evaluation results to {csv_out_path}")
    eval_result_conversations: List[List[Dict[str, Any]]] = eval_result.convos
    eval_df = pd.DataFrame(eval_result_conversations)
    eval_df.to_csv(csv_out_path, index=False)
