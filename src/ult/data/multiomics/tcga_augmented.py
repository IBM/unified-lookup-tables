"""Data loader for saved omics data in csv format."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2025
ALL RIGHTS RESERVED
"""
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict
from loguru import logger

from ..core import DataLoader

SPECIAL_CLASS_TOKEN_TO_INT_REGEX = re.compile("<C(\d+)>")


class MultiOmicsDataLoader(DataLoader):
    """Omics .csv data loader."""

    def __init__(
        self,
        data_dir: str,
        subsample: bool = False,
        subsample_fraction: float = 1e-1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize data loader.

        Args:
            data_dir: directory containing train and test data in .csv format
            subsample: whether to subsample data. Defaults to False.
            subsample_fraction: fraction of data used in subsampling. Defaults to 1e-1.
            standardise: whether to standardise data. Defaults to False.
            seed: seed used for random generation. Defaults to 42.
        """
        super().__init__(**kwargs)
        self.data_dir_path = Path(data_dir)
        self.train_csv_path = self.data_dir_path / "augmented_train_df_stdz.csv"
        self.test_csv_path = self.data_dir_path / "real_test_df_stdz.csv"
        self.train_labels_path = self.data_dir_path / "augmented_train_labels.csv"
        self.test_labels_path = self.data_dir_path / "real_test_labels.csv"
        self.subsample = subsample
        self.subsample_fraction = subsample_fraction
        self.seed = seed
        self.modality_keys = kwargs.get("modality_keys", None)
        self.rename_test = kwargs.get("rename_test", False)

    def load_dataset(self) -> Dataset | DatasetDict:
        """Load dataset.

        Returns:
            HF dataset or dataset dictionary.
        """
        datasets = {}
        for split, data_path, labels_path in [
            ("full_train", self.train_csv_path, self.train_labels_path),
            ("real_test", self.test_csv_path, self.test_labels_path),
        ]:
            # load into pandas data frames
            df = pd.read_csv(data_path, index_col=0)
            labels = pd.read_csv(labels_path, index_col=0)

            label_token_map = {
                label: f"<C{i}>" for i, label in enumerate(np.unique(labels.values))
            }

            with Path.open(self.data_dir_path / "label_map.json", "w") as f:
                json.dump(label_token_map, f)

            if self.modality_keys is None:
                df["array"] = pd.Series(df.values.tolist(), index=df.index).apply(
                    np.asarray
                )
            else:
                for key in self.modality_keys:
                    df[f"{key}_array"] = pd.Series(
                        df[df.columns[df.columns.str.startswith(key)]].values.tolist(),
                        index=df.index,
                    ).apply(np.asarray)

            merged_df = pd.concat(
                [df[df.columns[df.columns.str.contains("array")]], labels],
                axis=1,
                join="inner",
            )

            # NOTE: this line assumes the last column is the label column and renames it to target
            merged_df.rename(columns={merged_df.columns[-1]: "target"}, inplace=True)

            # process labels to encode as special tokens
            merged_df["target_token"] = merged_df["target"].replace(label_token_map)
            # using special tokens to get integer labels
            merged_df["target"] = [
                int(SPECIAL_CLASS_TOKEN_TO_INT_REGEX.match(target_token).group(1))
                for target_token in merged_df["target_token"]
            ]

            logger.info("Loaded all .csv files and renamed target.")

            hf_dataset = Dataset.from_pandas(merged_df)
            hf_dataset = hf_dataset.cast_column(
                "target", ClassLabel(num_classes=len(label_token_map.keys()))
            )
            datasets[split] = hf_dataset

        dataset_dict = DatasetDict(**datasets)

        return dataset_dict

    def load_splits(
        self, test_size: float | int = 0.1, val_size: float | int = 0.1, **kwargs: Any
    ) -> DatasetDict:
        """Load dataset in splits.

        Args:
            test_size: fraction of test sample or their absolute number. Defaults to 0.1.
                Unused as the dataset provide already splits.
            val_size: fraction of validation samples or their absolute number. Defaults to 0.1.
            **kwargs: additional arguments to pass to datasets.Dataset.train_test_split.

        Returns:
            HF dataset dict containing splits.
        """
        # load dataset
        dataset_dict = self.load_dataset()
        # split train, validation, test
        split_dataset = DatasetDict()
        dataset_train_test_split = dataset_dict["full_train"].train_test_split(
            test_size=test_size, seed=self.seed, stratify_by_column="target", **kwargs
        )
        dataset_train_val_split = dataset_train_test_split["train"].train_test_split(
            test_size=val_size, seed=self.seed, stratify_by_column="target", **kwargs
        )
        # val , train
        split_dataset["train"] = dataset_train_val_split["train"]
        split_dataset["val"] = dataset_train_val_split["test"]
        split_dataset["test"] = dataset_train_test_split["test"]
        split_dataset["real_test"] = dataset_dict["real_test"]

        if self.rename_test:
            split_dataset["synthetic_test"] = split_dataset.pop("test")
            split_dataset["test"] = split_dataset.pop("real_test")

        return split_dataset
