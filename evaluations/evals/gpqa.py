"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re

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


class GPQAEval(EvalBase):
    def __init__(
        self,
        variant: str = "diamond",
        num_examples: int
        | None = None,  # restrict to a subset of the data for debugging
        k_shots: int = 0,
    ):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
        self.examples = [row.to_dict() for _, row in df.iterrows()]

        rng = random.Random(0)

        self.examples = [
            example | {"permutation": rng.sample(range(4), 4)}
            for example in self.examples
        ]
        self.examples = self.add_k_shot_samples(self.examples, k_shots)

        if num_examples:
            self.examples = rng.sample(self.examples, num_examples)

    def make_message(self, row: dict) -> dict:
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in row["permutation"]]
        choices_dict = dict(
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            Question=row["Question"],
        )
        return dict(
            content=format_multichoice_question(choices_dict),
            role="user",
        )

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            k_shot_questions = [self.make_message(row) for row in row["k_shots"]]
            k_shot_answers = [
                dict(content=row["Correct Answer"], role="assistant")
                for row in row["k_shots"]
            ]
            k_shot_messages = [
                message
                for pair in zip(k_shot_questions, k_shot_answers)
                for message in pair
            ]
            prompt_messages = k_shot_messages + [self.make_message(row)]
            correct_answer = row["Correct Answer"]

            response_text = sampler(prompt_messages)

            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == correct_answer else 0.0

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                score=score,
                convo=convo,
                metrics={"chars": len(response_text)},
            )

        results = map_with_progress(fn, self.examples)
        return aggregate_results(results)
