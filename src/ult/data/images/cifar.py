"""Cifar10 data loader for image data."""

from typing import Any, Optional

from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset

from ..core import DataLoader


class CifarImageDataLoader(DataLoader):
    """Data Loader class to load the Cifar images datasets."""

    def __init__(
        self,
        do_shuffle: Optional[bool] = False,
        num_proc: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Init Cifar Loader

        Args:
            do_downsample: Downsample dataset.
            do_shuffle: Shuffle dataset.
            downsample_size: Size to downsample dataset.
        """
        self.num_proc = num_proc
        self.do_shuffle = do_shuffle
        self.additional_kwargs = kwargs
        

    def load_dataset(self) -> DatasetDict | Dataset:
        """Load Cifar dataset

        Returns:
            An DatasetDict with train, val and test splits.
        """

        # Load and create train val split
        download_config = DownloadConfig(
            delete_extracted=True, resume_download=True, num_proc=self.num_proc
        )
        dataset = load_dataset(
            "uoft-cs/cifar10",
            num_proc=self.num_proc,
            download_config=download_config,
            **self.additional_kwargs,
        )

        # shuffle
        if self.do_shuffle:
            dataset = dataset.shuffle()

        # from PIL to np
        dataset = dataset.with_format(type="numpy", columns=["img"], output_all_columns=True)

        # rename column
        dataset = dataset.rename_column("img", "pix_array")
        dataset = dataset.rename_column("label", "labels")

        # create validation split and final dataset
        dataset_train_val_split = dataset["train"].train_test_split(
            test_size=0.1, stratify_by_column="labels"
        )
        dataset_split = DatasetDict()
        dataset_split["test"] = dataset["test"]
        dataset_split["train"] = dataset_train_val_split["train"]
        dataset_split["val"] = dataset_train_val_split["test"]

        return dataset_split
