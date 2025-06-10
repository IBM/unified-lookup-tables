"""Generic evaluator."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
import importlib
import logging
import re
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset
from transformers import EvalPrediction

from .core import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# NOTE: Since we pass function as string, we need to ensure the package is imported. Here, torchmetrics.
class GenericEvaluator(Evaluator):
    """This class can be used for single or multi-target evaluation.

    Args:
        Evaluator: Base class.
    """

    def __init__(
        self,
        prediction_target_to_task_type: Dict[str, str],
        prediction_target_to_groundtruth_target: Dict[str, str],
        processing_pipeline_operators_per_target: Dict[str, Any],
        grouped_metrics_dict: str,
        **kwargs: Any,
    ) -> None:
        """Initialise GenericEvaluator.

        Args:
            prediction_target_to_task_type: prediction target to task type.
            prediction_target_to_groundtruth_target: mapping between prediction and groundtruth target.
            processing_pipeline_operators_per_target: post-processor for target (here the key is a prediction target).
            grouped_metrics_dict: metric functions (values) to apply to task types (keys).
        """
        super().__init__(**kwargs)
        self.prediction_target_to_task_type = prediction_target_to_task_type
        self.prediction_target_to_groundtruth_target = prediction_target_to_groundtruth_target
        self.groundtruth_target_to_prediction_target = {
            groundtruth_target: prediction_target
            for prediction_target, groundtruth_target in self.prediction_target_to_groundtruth_target.items()
        }
        self.prediction_fields, self.groundtruth_fields = [
            cast(List[str], elements)
            for elements in zip(*prediction_target_to_groundtruth_target.items())
        ]

        self.task_type_to_metric_fns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for _, task_type in self.prediction_target_to_task_type.items():
            task_type_metrics = eval(grouped_metrics_dict).get(task_type, [])
            for metric in task_type_metrics:
                package_name, package_object = GenericEvaluator.import_package(metric)
                if package_name and package_object:
                    package_name = (
                        getattr(package_object, "metric_name")
                        if hasattr(package_object, "metric_name")
                        else package_name
                    )
                    self.task_type_to_metric_fns[task_type][package_name] = package_object

        self.processing_pipeline_operators_per_target = processing_pipeline_operators_per_target

    @staticmethod
    def import_package(package_descriptor: str) -> Tuple[Optional[str], Optional[Any]]:
        """Import a package.

        Args:
            package_descriptor: name of the package to import and its arguments.

        Returns:
            The package as object and instantiated package object.
        """
        try:
            package_module: str = ".".join(package_descriptor.split(".")[:-1])
            package_name: str = package_descriptor.split(".")[-1].split("(")[0]
            pattern = r"(\w+)\s*=\s*([^,|)$]*)"  # pattern to find arguments like '(arg=val)'
            package_arguments: Dict[str, Any] = {
                str(match[0]): eval(match[1]) for match in re.findall(pattern, package_descriptor)
            }
            package_object = getattr(importlib.import_module(package_module), package_name)

            # if package_object is a callable just package_object(**package_arguments) will trigger __call__
            # and raise an error as it expects package_object(predictions, target)
            if isinstance(package_object, types.FunctionType):
                instantiated_package_object = package_object
            else:
                # NOTE: we assume the object is a class implementing __call__
                instantiated_package_object = package_object(**package_arguments)

        except ImportError as import_error_exception:
            logger.warning(
                f"missing dependencies present, make sure to install them (details: {import_error_exception})"
            )
            instantiated_package_object = None
        return package_name, instantiated_package_object

    def apply_postprocessing(
        self, generated_model_test_output: Any
    ) -> Dict[str, List[Any] | torch.Tensor]:
        """Applies postprocessing functions to predictions.

        Args:
            generated_model_test_output: predictions from the model.

        Returns:
            Processed predictions for metric computations.
        """
        # NOTE: the generic evaluator covers the postprocessing case where
        # we use the model predictions as-is. Useful for text based models.
        return {
            target: cast(List[Any], generated_model_test_output)
            for target in self.prediction_fields
        }

    def get_prediction_groundtruth_to_evaluate(
        self,
        relevant_prediction: (List[str | npt.NDArray[np.float32]] | npt.NDArray[np.float32]),
        relevant_groundtruth: List[str | torch.Tensor] | npt.NDArray[np.float32],
    ) -> Tuple[
        torch.Tensor | List[str | float | List[float]],
        torch.Tensor | List[str],
    ]:
        """Converts predictions and groundtruth into the appropiate format. String: no change, Tensor[size=1]: float, Tensor[size=n]: Tensor[size=n].

        Args:
            relevant_prediction: Predictions
            relevant_groundtruth: Groundtruth

        Returns:
            Tuple containing predictions and groundtruth

        """

        prediction_to_evaluate_list: List[float | str | List[float]] = []
        predictions_contain_strings: bool = True
        for prediction in relevant_prediction:
            if (
                isinstance(prediction, (np.ndarray, torch.Tensor))
                and prediction.flatten().shape[0] == 1
            ):
                prediction_to_evaluate_list.append(prediction.item())
                predictions_contain_strings = False
            elif isinstance(prediction, (np.ndarray, torch.Tensor)):
                prediction_to_evaluate_list.append(prediction.tolist())
                predictions_contain_strings = False
            else:
                prediction_to_evaluate_list.append(prediction)

        prediction_to_evaluate: torch.Tensor | List[str | float | List[float]]
        if not predictions_contain_strings:
            prediction_to_evaluate = torch.Tensor(prediction_to_evaluate_list)
            groundtruth_to_evaluate = torch.Tensor(relevant_groundtruth[: len(relevant_prediction)])
        else:
            prediction_to_evaluate = prediction_to_evaluate_list
            groundtruth_to_evaluate = relevant_groundtruth[: len(relevant_prediction)]  # type: ignore
        return prediction_to_evaluate, groundtruth_to_evaluate

    def compute_metrics_training(self, eval_preds: EvalPrediction) -> Dict[str, Dict[str, Any]]:
        """Computes metrics during training.

        Much simpler as during training we don't have access to names of predictions/labels.
        Assumes only one prediction/target field and one ndarray for prediction and one ndarray for labels

        Args:
            eval_preds: EvalPrediction generated during evaluation_step

        Returns:
            metrics computed.
        """

        predictions = self.apply_postprocessing(eval_preds.predictions)
        # NOTE: the generic evaluator covers the case where we use the same groundtruth
        # for all targets
        groundtruth = {target: eval_preds.label_ids for target in self.groundtruth_fields}
        metrics = self._compute_metrics(predictions=predictions, groundtruth=groundtruth)
        flattened_metrics_dict = {
            f"{outer_key}_{inner_key}": inner_value
            for outer_key, outer_dict in metrics.items()
            for inner_key, inner_value in outer_dict.items()
        }
        return flattened_metrics_dict

    def compute(
        self,
        generated_model_test_output: List[str | torch.Tensor | npt.NDArray[np.float32]],
        groundtruth: Dataset,
    ) -> Dict[str, Dict[str, Any]]:
        """Computes metrics only.

        Args:
            generated_model_test_output: Predictions as generated by model.
            groundtruth: groundtruth values of the predictions.

        Returns:
            metrics computed.
        """

        predictions = self.apply_postprocessing(generated_model_test_output)
        return self._compute_metrics(predictions=predictions, groundtruth=groundtruth)

    def _compute_metrics(
        self,
        predictions: Dict[str, List[Any | torch.Tensor] | torch.Tensor],
        groundtruth: Dataset | Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metrics predictions and groundtruth.

        Args:
            predictions: predictions to evaluate.
            groundtruth: groundtruth for evaluation.

        Returns:
            metrics computed.
        """
        metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for target in self.groundtruth_fields:
            prediction_target = self.groundtruth_target_to_prediction_target[target]
            task_type = self.prediction_target_to_task_type[prediction_target]
            if any([pred is None for pred in predictions[prediction_target]]):
                logger.warning(
                    f"predictions for {target} include None. This target will be skipped from the evaluation."
                )
            else:
                for metric_name, metric_fn in self.task_type_to_metric_fns[task_type].items():
                    metric_key_for_target = f"{target}_{metric_name}"
                    prediction_to_evaluate, groundtruth_to_evaluate = (
                        self.get_prediction_groundtruth_to_evaluate(
                            predictions[prediction_target],  # type: ignore[arg-type]
                            groundtruth[target],
                        )
                    )
                    try:
                        computed_metric = metric_fn(
                            prediction_to_evaluate,
                            groundtruth_to_evaluate,
                        )
                    except ZeroDivisionError:
                        # NOTE: in case of zero divison error we return nan since it is a valid float
                        computed_metric = np.nan

                    if isinstance(computed_metric, dict):
                        for key, value in computed_metric.items():
                            # NOTE: when possible we try to cast the value to float
                            try:
                                metrics[task_type][f"{metric_key_for_target}_{key}"] = float(value)
                            except (ValueError, TypeError):
                                # NOTE: here we assume if a value has an attribute tolist
                                # then it contains primitive types and it will be JSON serializable
                                # once converted.
                                if hasattr(value, "tolist"):
                                    metrics[task_type][f"{metric_key_for_target}_{key}"] = (
                                        value.tolist()
                                    )
                                else:
                                    metrics[task_type][f"{metric_key_for_target}_{key}"] = value
                    else:
                        metrics[task_type][metric_key_for_target] = float(computed_metric)
        return metrics


class AutoRegressiveClassificationEvaluator(GenericEvaluator):
    """Evaluator for classification tasks where labels are generated autoregressively.
    Args:
        GenericEvaluator: generic evaluator class.
    """

    def __init__(
        self,
        prediction_target_to_task_type: Dict[str, str],
        prediction_target_to_groundtruth_target: Dict[str, str],
        processing_pipeline_operators_per_target: Dict[str, Any],
        grouped_metrics_dict: str,
        num_classes: int,
        postprocessing_function_str: str,
        evaluation_storage_path: Path,
        **kwargs: Any,
    ) -> None:
        """Initialise Generic Evaluator for Autoregressive Classification.

        Args:
            postprocessing_function: prediction target to task type.
            prediction_target_to_groundtruth_target: mapping between prediction and groundtruth target.
            processing_pipeline_operators_per_target: post-processor for target (here the key is a prediction target).
            grouped_metrics_dict: metric functions (values) to apply to task types (keys).
            num_classes: number of classes.
            postprocessing_function_str: string with the post-processing function to apply to each generated output.
        """

        super().__init__(
            prediction_target_to_groundtruth_target=prediction_target_to_groundtruth_target,
            prediction_target_to_task_type=prediction_target_to_task_type,
            processing_pipeline_operators_per_target=processing_pipeline_operators_per_target,
            grouped_metrics_dict=grouped_metrics_dict,
            **kwargs,
        )

        self.postprocessing_function = lambda example: eval(postprocessing_function_str)  # noqa
        # NOTE: valid classes will be 0..num_classes-1 and the last one num_classes is left for class parsings that fail
        self.num_classes = num_classes + 1
        self.evaluation_storage_path = evaluation_storage_path

    def apply_postprocessing(
        self,
        generated_model_test_output: List[str | torch.Tensor | npt.NDArray[np.float32]],
    ) -> Dict[str, List[Any | torch.Tensor] | torch.Tensor]:
        """Applies sigmoid function to predictions.

        Args:
            generated_model_test_output: predictions from the model.

        Returns:
            Processed predictions for metric computations.
        """

        postprocessing_targets: Dict[
            str, List[Any | torch.Tensor] | torch.Tensor
        ] = {}  # class labels should be integers

        for target in self.prediction_fields:
            list_target_values = []
            for sample in generated_model_test_output:
                try:
                    target_value = self.postprocessing_function(sample)
                    if target_value not in range(self.num_classes - 1):
                        target_value = self.num_classes  # assign invalid class
                except:  # noqa
                    target_value = self.num_classes  # assign invalid class

                list_target_values.append(target_value)

            postprocessing_targets[target] = torch.tensor(list_target_values)

        return postprocessing_targets

    def compute(
        self,
        generated_model_test_output: List[str | torch.Tensor | npt.NDArray[np.float32]],
        groundtruth: Dataset,
    ) -> Dict[str, Dict[str, Any]]:
        """Computes metrics only.

        Args:
            generated_model_test_output: Predictions as generated by model.
            groundtruth: groundtruth values of the predictions.

        Returns:
            metrics computed.
        """

        predictions = self.apply_postprocessing(generated_model_test_output)

        return self._compute_metrics(predictions=predictions, groundtruth=groundtruth)
