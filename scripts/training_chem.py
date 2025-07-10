"""Run the training pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from pathlib import Path

import click
from omegaconf import OmegaConf

from ult.modeling.training_chem import TrainPipeline
from scale_training.modeling.utils import seed_everything


@click.command()
@click.option(
    "--pipeline_configuration_path",
    help="path to the yaml file defining the pipeline to be executed",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--seed",
    help="Seed for randomicity",
    default=42,
    type=int,
)
def execute_training_pipeline(pipeline_configuration_path: Path, seed: int) -> None:
    """Execute the training pipeline."""

    seed_everything(seed)

    config = OmegaConf.load(pipeline_configuration_path)
    model_name = config["model_name"]
    dataset_location = config["dataset_location"]
    dataset_name = config["dataset_name"]
    
    training_args = OmegaConf.to_container(config["training_args"], resolve=True)
    kwargs = config["kwargs"]

    train_pipeline = PIPELINE_REGISTRY[model_name](
        model_name=model_name,
        dataset_location=dataset_location,
        dataset_name=dataset_name,
        training_args=training_args,
        seed=seed,
        **kwargs,
    )
    train_pipeline.run_training_pipeline()


if __name__ == "__main__":
    execute_training_pipeline()
