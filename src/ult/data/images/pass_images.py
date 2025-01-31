"""PASS smiles Dataset Loader."""

from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, DownloadConfig, Image, load_dataset
from PIL import PngImagePlugin

from ..core import DataLoader

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (128 * 128 * 3)


class PASSImageDataLoader(DataLoader):
    """Data Loader class to load the PASS images dataset"""

    def __init__(
        self,
        do_downsample: Optional[bool] = False,
        do_shuffle: Optional[bool] = False,
        downsample_factor: Optional[float] = None,
        do_rescale: Optional[bool] = None,
        rescale_size: Optional[List[int]] = None,
        num_proc: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Init PASS Loader

        Args:
            do_downsample: Downsample dataset.
            downsample_factor: Downsample factor.
            do_shuffle: Shuffle dataset.
            do_rescale: Rescale dataset.
            rescale_size: Rescale image size.
            downsample_size: Size to downsample dataset.
        """
        self.do_shuffle = do_shuffle
        self.do_downsample = do_downsample
        self.downsample_factor = downsample_factor
        self.num_proc = num_proc
        self.do_rescale = do_rescale
        self.rescale_size = rescale_size
        self.additional_kwargs = kwargs

    def load_dataset(self) -> DatasetDict | Dataset:
        """Load PASS method

        Returns:
            An DatasetDict with train, val and test splits.
        """

        # Load and create train val split

        download_config = DownloadConfig(
            delete_extracted=True, resume_download=True, num_proc=self.num_proc
        )
        dataset = load_dataset(
            "yukimasano/pass",
            num_proc=self.num_proc,
            split="train",
            download_config=download_config,
            **self.additional_kwargs,
        )

        # Remove unused columns
        dataset = dataset.remove_columns(
            [
                "creator_username",
                "hash",
                "gps_latitude",
                "gps_longitude",
                "date_taken",
            ]
        )

        # Shuffle
        if self.do_shuffle:
            dataset = dataset.shuffle()

        # Subsample
        if self.do_downsample and self.downsample_factor is not None:
            dataset = dataset.select(range(int(len(dataset) * self.downsample_factor)))

        dataset = dataset.filter(
            lambda example: example["image"].mode == "RGB", num_proc=self.num_proc
        )

        # Rescale size images
        if self.do_rescale:

            def transforms(examples: Dict[str, Any]) -> Dict[str, Any]:
                examples["image"] = [
                    image.resize(self.rescale_size) for image in examples["image"]
                ]
                return examples

            dataset = dataset.map(transforms, batched=True, num_proc=self.num_proc)

        # convert to RGB
        dataset = dataset.cast_column("image", Image(mode="RGB"))

        # from PIL to np
        dataset = dataset.with_format(
            type="numpy", columns=["image"], output_all_columns=True
        )

        # rename column
        dataset = dataset.rename_column("image", "pix_array")

        return dataset
