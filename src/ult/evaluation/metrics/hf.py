"""Hugging Face evaluate metrics."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from typing import Any, List, Union

import evaluate
import torch


class HuggingFaceMetricWrapper:
    """Hugging Face evaluate metric wrapper."""

    def __init__(self, evaluator_name: str, **kwargs: Any) -> None:
        """Initialize the metric wrapper.

        Args:
            evaluator_name: evaluator name.
        """
        self.evaluator_name = evaluator_name
        self.metric_name = f"HF{evaluator_name.title()}"
        self.evaluator_compute_kwargs = kwargs
        self.evaluator = evaluate.load(evaluator_name)

    def __call__(
        self,
        predictions: Union[List[str], torch.Tensor],
        groundtruth: Union[List[str], torch.Tensor],
    ) -> Any:
        """Run the metric computation.

        Args:
            predictions: predictions.
            groundtruth: groundtruth.

        Returns:
            the metric computation result.
        """
        return self.evaluator.compute(
            predictions=predictions,
            references=groundtruth,
            **self.evaluator_compute_kwargs,
        )
