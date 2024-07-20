import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class EvalBase(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def sample_k_shots(
        self, row: Dict[str, Any], k_shots: int, examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        sample = random.sample(examples, k_shots)
        # Assert that the sample does not contain the row itself
        while row in sample:
            sample.remove(row)
            sample.append(random.choice(examples))
        return sample

    def add_k_shot_samples(
        self, examples: List[Dict[str, Any]], k_shots: int
    ) -> List[Dict[str, Any]]:
        k_shot_samples_per_example = [
            self.sample_k_shots(row, k_shots, examples) for row in examples
        ]
        return [
            {**row, "k_shots": k_shot_samples}
            for row, k_shot_samples in zip(examples, k_shot_samples_per_example)
        ]
