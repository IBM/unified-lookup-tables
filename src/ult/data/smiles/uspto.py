"""USPTO smiles Dataset Loader."""

from pathlib import Path

import pandas as pd
import rdkit.Chem as Chem
from datasets import Dataset, DatasetDict
from rdkit import RDLogger

from ..core import DataLoader


def canonicalize(smiles: str) -> str:
    """Helper function to canonicalise smiles strings.

    Args:
        smiles: smiles string.

    Returns:
        Canonicalised string or None if invalid smiles.
    """
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "C.C"  # NOTE: returning dummy string instead of None to not stop metric computation

    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]  # type: ignore[call-arg,no-untyped-call]
    return Chem.MolToSmiles(mol)


class SmilesUSPTODataLoader(DataLoader):
    """Recipes data loader."""

    def __init__(
        self,
        smiles_csv_path: Path,
        subsample: bool = False,
        subsample_fraction: float = 0.001,
        seed: int = 42,
    ) -> None:
        """Init smiles loader class."""
        self.smiles_csv_path = smiles_csv_path
        self.subsample = subsample
        self.subsample_fraction = subsample_fraction
        self.seed = seed

    def load_dataset(self) -> Dataset | DatasetDict:
        """Load dataset.

        Returns:
            The Dataset object with loaded UPSTO data.
        """

        dataset_split = DatasetDict()

        for split in ["train", "val", "test"]:
            file = Path(self.smiles_csv_path) / f"{split}.txt"
            if file.is_file():
                # load
                pd_smiles_split = pd.read_csv(
                    file, delimiter=">>", names=["reagents", "products"]
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

            # assign split
            dataset_split[split] = Dataset.from_pandas(pd_smiles_split)

        return dataset_split
