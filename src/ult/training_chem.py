__copyright__ = """LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2025 - AI4SD
ALL RIGHTS RESERVED
"""

import contextlib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, TextIO, cast, Optional

import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset, concatenate_datasets
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from loguru import logger
from multiprocessing import cpu_count
import numpy as np

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from scale_training.modeling.utils import StreamToLogger, seed_everything

IS_CUDA_AVAILABLE = torch.cuda.is_available()


device_string = PartialState().process_index


class TrainPipeline:
    """Class to finetune CausalLMs with LoRA.

    Attributes
    ----------
    model_name: str
        Model name on HF.
    dataset_location: str
        Either local or HF dataset location.
    lora_config: Dict[str, Any]
        Configuration for LoRA adapter.
    training_args: Dict[str, Any]
        Training Arguments for HF Trainer.
    **kwargs


    Methods
    -------
    create_prompt()
        Generate the instruction tuning prompt.
    get_dataset()
        Load the instruction tuning dataset.
    load_model()
        Load the HF model from the hub.
    setup_lora()
        Setup the LoRA adapter.
    run_training_pipeline()
        Setup and run the training pipeline.
    """

    def __init__(
        self,
        model_name: str,
        dataset_location: str,
        dataset_name: Optional[str],
        training_args: Dict[str, Any],
        seed: int,
        **kwargs,
    ):
        """Initialise configs for training pipeline."""
        self.model_name = model_name
        self.dataset_location = Path(dataset_location)
        self.training_args = training_args
        self.seed = seed
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            self.date = [datetime.now().strftime("%Y-%m-%d-%H-%M-%S")]
        else:
            self.date = [None]

        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(self.date, 0)

        self.date = self.date[0]

        # logger.info(f"Rank [{torch.distributed.get_rank()}]: date: {self.date}")

        self.keep_in_memory = False
        if IS_CUDA_AVAILABLE and torch.cuda.device_count() > 1:
            self.keep_in_memory = True

        logger.remove()
        logger.add(cast(TextIO, sys.__stderr__), enqueue=True)
        if self.training_args["logging_dir"]:
            logger.add(
                Path(self.training_args["logging_dir"])
                / f"{self.model_name.split('/')[-1]}-{self.date}.log",
                enqueue=True,  # NOTE: added to ensure log file is written with no lock during distributed training
            )
        self.stream = StreamToLogger(level="INFO")
        # logger = self.init_logger()

        if not IS_CUDA_AVAILABLE:
            torch.backends.mps.is_available = lambda: False  # type:ignore

    def fail_safe_conditional_distributed_barrier(self, condition_fn: Callable[[], bool]) -> None:
        """Apply a distributed barrier in a fail-safe way.

        Args:
            condition_fn: callable to define condition for the barrier.
        """
        try:
            if condition_fn():
                logger.info("Distributed barrier applied")
                torch.distributed.barrier()
                logger.info("Barrier ended.")
        except ValueError:
            # NOTE: catching errors due to uninitialized distributed process group.
            # Never active when running without torchrun. In this case a barrier is never needed.
            logger.info("No distributed barrier applied")


    def load_chem_dataset(
        self,
        fname: str = "train.jsonl",
        include_task_name: bool = False,
        subsample: float = 1.0,
        cpu_cores: int = 8,
    ) -> Dataset:
        """
        Loads chemistry datasets. Expect a folder with one subfolder per task (folder name is taskname).
        Each subfolder contains a jsonl file with a given filename.

        Args:
            data_dir: Path to the folder containing the data
            fname: filename e.g. train.jsonl
            include_task_name: Wether to add a column specifying the task name to the dataset or not
            subsample: Fraction of dataset to use. Defaults to 1.0 -> Full dataset
            cpu_cores: Number of cpus to use during processing
        Returns:
            Dataset loaded from all folders.
        """

        datasets = list()

        for folder in self.dataset_location.glob("*"):
            dataset_name = folder.name
            if not folder.is_dir():
                logger.warning(
                    f"Found unexpected file: {dataset_name} in folder: {self.dataset_location}"
                )
                continue

            logger.info(f"Loading dataset {dataset_name}")

            if not (folder / fname).is_file():
                logger.warning(
                    f"Could not find file: {fname} in folder: {folder}. Skipping dataset {dataset_name}"
                )
                continue

            dataset = load_dataset(
                "json",
                data_files={fname.split(".")[0]: str(folder / fname)},
                split=fname.split(".")[0],
                streaming=False,
                keep_in_memory=False,
                cache_dir=os.getenv("HF_DATASETS_CACHE"),
            )

            if include_task_name:
                dataset = dataset.map(
                    lambda _: {"task_name": dataset_name}, num_proc=cpu_cores
                )

            # Subsample on individual dataset level: retain the distributions between datasets
            selected_idx = np.random.choice(
                len(dataset), int(subsample * len(dataset)), replace=False
            )
            subsampled_dataset = dataset.select(selected_idx)

            datasets.append(subsampled_dataset)

        complete_dataset = concatenate_datasets(datasets)

        return complete_dataset

    def load_model(self):
        """Load the HF model from the hub."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        #if self.model_name == "ibm-granite/granite-3.1-1b-a400m-instruct":
        #    model.config._attn_implementation = "flash_attention_2"

        logger.info("Number of trainable params before LoRA")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Model: {self.model_name} successfully initialised for rank: {device_string}.")
        estimate_zero3_model_states_mem_needs_all_live(model)

        return model, tokenizer



    def run_training_pipeline(self):
        """Setup and run the training pipeline."""
        with contextlib.redirect_stderr(self.stream):  # type:ignore
            try:
                if not torch.distributed.is_initialized():
                    try:
                        torch.distributed.init_process_group(
                            backend="nccl" if IS_CUDA_AVAILABLE else "gloo",
                            timeout=timedelta(
                                minutes=float(
                                    os.getenv("TORCH_PROCESS_GROUP_TIMEOUT_IN_MINUTES", 30)
                                )
                            ),
                        )
                        logger.info("Process group has been initialized successfully")
                    except ValueError:
                        logger.warning(
                            "Initializing the process group from the environment was not possible!"
                        )

                seed_everything(self.seed)

                self.fail_safe_conditional_distributed_barrier(
                    lambda: torch.distributed.get_rank() > 0
                )

                model, tokenizer = self.load_model()

                dataset = self.load_chem_dataset()
                
                def tokenize_fn(example):
                    text = example['source'] + example['target']
                    return tokenizer(text, truncation=True, padding=False, max_length=1024)

                reduced_dataset = dataset.select(np.random.choice(len(dataset), 400000, replace=False))
                tokenized_dataset = reduced_dataset.map(tokenize_fn, batched=False, remove_columns=["source", "target"], num_proc=8)
                
                selected_samples = list()
                n_tokens = 0
                for i, sample in enumerate(tokenized_dataset):
                    n_tokens += len(sample['input_ids'])
                    selected_samples.append(i)

                    if n_tokens > 5e6:
                        break

                logger.info(f"Dataset consists of {len(selected_samples)} samples with {n_tokens} tokens.")

                dataset_50M = tokenized_dataset.select(selected_samples)

                train_val_test = dataset_50M.train_test_split(test_size=0.2, seed=self.seed)

                if self.training_args["resume_from_checkpoint"] is not None:
                    model_identifier = Path(self.training_args["resume_from_checkpoint"]).name
                else:
                    model_identifier = f"{self.model_name.split('/')[-1]}-{self.date}"

                logger.info(f"Model id: {model_identifier}")

                self.training_args["output_dir"] = (
                    f"{self.training_args['output_dir']}/{model_identifier}"
                )
                self.training_args["logging_dir"] = (
                    f"{self.training_args["output_dir"]}/{logs}"
                )

                train_dataset = train_val_test["train"]
                validation_set = train_val_test["test"]

                logger.info(f"Len of the dataset: {len(train_dataset)}")

                self.fail_safe_conditional_distributed_barrier(
                    lambda: torch.distributed.get_rank() == 0 and IS_CUDA_AVAILABLE
                )

                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False
                )

                logger.info("Initialise trainer")
                trainer = Trainer(
                    model=model,
                    args=TrainingArguments(**self.training_args),
                    train_dataset=train_dataset,
                    data_collator=data_collator,
                    processing_class=tokenizer,
                )

                self.fail_safe_conditional_distributed_barrier(lambda: True)
                logger.info("Starting training")
                trainer.train()
                logger.info("Saving best model.")

                trainer.save_model(f"{self.training_args['output_dir']}/{model_identifier}-final")
            except Exception:
                logger.exception("Pipeline execution failed!")
