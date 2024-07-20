"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re
from typing import Any, Dict, List, Optional
from enum import Enum
import pandas

from common import (
    ANSWER_PATTERN_MULTICHOICE,
    aggregate_results,
    format_multichoice_question,
    map_with_progress,
)
from evals.base import EvalBase
from models.base import SamplerBase
from typings import EvalResult, SingleEvalResult

class EvaluationType(Enum):
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    MULTIPLICATION = "multiplication"
    DIVISION = "division"

    def from_str(s: str) -> "EvaluationType":
        if s == "addition":
            return EvaluationType.ADDITION
        elif s == "subtraction":
            return EvaluationType.SUBTRACTION
        elif s == "multiplication":
            return EvaluationType.MULTIPLICATION
        elif s == "division":
            return EvaluationType.DIVISION
        else:
            raise ValueError(f"Invalid EvaluationType: {s}")

DATA_FILE_PATH = "./aritheval/numbers.csv"

class ArithEvalBase(EvalBase):
    def __init__(
            self,
            eval_type: EvaluationType,
            num_examples: Optional[int] = None,
        ) -> None:
        # Read in the CSV
        df = pandas.read_csv(DATA_FILE_PATH)
        self.examples = df.to_dict(orient="records")
        if num_examples is not None:
            self.examples = random.sample(self.examples, num_examples)
        self.eval_type = eval_type

    def get_prompt(self, number1: int, number2: int, eval_type: EvaluationType) -> str:
        """
        Generate a prompt from two numbers and the evaluation type.
        """
        prompt = ""
        if eval_type == EvaluationType.ADDITION:
            prompt = f"What is {number1} plus {number2}?"
        elif eval_type == EvaluationType.SUBTRACTION:
            prompt = f"What is {number1} minus {number2}?"
        elif eval_type == EvaluationType.MULTIPLICATION:
            prompt = f"What is {number1} times {number2}?"
        elif eval_type == EvaluationType.DIVISION:
            prompt = f"What is {number1} divided by {number2}?"

        prompt += "Think step by step and return your answer as Answer: <answer>."
        return prompt

    def get_target(self, number1: int, number2: int, eval_type: EvaluationType) -> str:
        """
        Get the target from two numbers and the evaluation type.
        """
        if eval_type == EvaluationType.ADDITION:
            return str(number1 + number2)
        elif eval_type == EvaluationType.SUBTRACTION:
            return str(number1 - number2)
        elif eval_type == EvaluationType.MULTIPLICATION:
            return str(number1 * number2)
        elif eval_type == EvaluationType.DIVISION:
            return str(number1 / number2)

    def evaluate_generation(self, generation: str, target: str) -> float:
        """
        If the target is present in the generation, we return 1.0, otherwise we return 0.0.
        """
        return 1.0 if target in generation else 0.0

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def get_single_eval_result(row: Dict[str, Any]) -> SingleEvalResult:

            prompt = self.get_prompt(row["number1"], row["number2"], self.eval_type)
            prompt_messages = sampler._pack_message(
                content=prompt, role="user"
            )
            response_text = sampler(prompt_messages)

            if len(response_text.strip()) == 0:
                score = 0.0
            else:
                target = self.get_target(row["number1"], row["number2"], self.eval_type)
                score = self.evaluate_generation(response_text, target)

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(score=score, convo=convo)

        results = map_with_progress(get_single_eval_result, self.examples)
        return aggregate_results(results)

class AdditionEval(ArithEvalBase):
    def __init__(self, num_examples: Optional[int] = None) -> None:
        super().__init__(eval_type=EvaluationType.ADDITION, num_examples=num_examples)

class SubtractionEval(ArithEvalBase):
    def __init__(self, num_examples: Optional[int] = None) -> None:
        super().__init__(eval_type=EvaluationType.SUBTRACTION, num_examples=num_examples)

class MultiplicationEval(ArithEvalBase):
    def __init__(self, num_examples: Optional[int] = None) -> None:
        super().__init__(eval_type=EvaluationType.MULTIPLICATION, num_examples=num_examples)

class DivisionEval(ArithEvalBase):
    def __init__(self, num_examples: Optional[int] = None) -> None:
        super().__init__(eval_type=EvaluationType.DIVISION, num_examples=num_examples)