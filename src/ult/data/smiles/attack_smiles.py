"""USPTO smiles Dataset Loader."""

from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset, DatasetDict

from ..core import DataLoader
from .uspto import canonicalize


class AttackSmilesDataLoader(DataLoader):
    """Recipes data loader."""

    def __init__(
        self,
        smiles_csv_path: Path,
        split: List[str] = ["train"],
        subsample: bool = False,
        subsample_fraction: float = 0.001,
        seed: int = 42,
    ) -> None:
        """Init smiles loader class."""
        self.smiles_csv_path = Path(smiles_csv_path)
        self.subsample = subsample
        self.subsample_fraction = subsample_fraction
        self.seed = seed
        self.split = split

    def load_dataset(self) -> Dataset | DatasetDict:
        """Load dataset.

        Returns:
            The Dataset object with loaded UPSTO data.
        """

        dataset_split = DatasetDict()

        for split in self.split:
            assert self.smiles_csv_path.suffix == ".csv", print(
                RuntimeError("File is not a csv")
            )
            pd_smiles_split = pd.read_csv(
                self.smiles_csv_path, usecols=["reagents", "products"]
            )
            # subsample
            if self.subsample:
                pd_smiles_split = pd_smiles_split.sample(
                    frac=self.subsample_fraction, random_state=self.seed
                )

            # canonicalisation
            pd_smiles_split["reagents"] = pd_smiles_split["reagents"].apply(
                canonicalize
            )
            pd_smiles_split["products"] = pd_smiles_split["products"].apply(
                canonicalize
            )

            # assign split - bug: if not is_file, then will raise an error as pd_smiles_split isnt defined
            dataset_split[split] = Dataset.from_pandas(pd_smiles_split)

        return dataset_split
