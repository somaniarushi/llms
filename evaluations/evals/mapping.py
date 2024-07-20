from evals.aritheval import (
    AdditionEval,
    DivisionEval,
    MultiplicationEval,
    SubtractionEval,
)
from evals.drop import DropEval
from evals.gpqa import GPQAEval
from evals.mmlu import MMLUEval
from typings import EvalResult

EVAL_TASK_MAPPING = {
    "mmlu": MMLUEval,
    "drop": DropEval,
    "gpqa": GPQAEval,
    "addition": AdditionEval,
    "subtraction": SubtractionEval,
    "multiplication": MultiplicationEval,
    "division": DivisionEval,
}


def run_eval_from_name(
    eval_name: str,
    server: str,
    num_examples: int,
    k_shots: int,
) -> EvalResult:
    eval_class = EVAL_TASK_MAPPING[eval_name]
    eval_instance = eval_class(num_examples=num_examples, k_shots=k_shots)
    return eval_instance(server)
