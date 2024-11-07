"""Text Dataset Loader."""

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from ..core import DataLoader


class WikiText103Loader(DataLoader):
    """Wikitext 103 data loader from HF."""

    def __init__(self, wiki_config: str = "wikitext-2-raw-v1", **kwargs: Any) -> None:
        """Initialise WikiText103Loader."""
        super().__init__(**kwargs)
        self.wiki_config = wiki_config
        self.split_functions = kwargs.get("split_functions", None)

    def load_dataset(self) -> Dataset | DatasetDict:
        """Load dataset.

        Returns:
            HF dataset dictionary of Wikitext103.
        """

        dataset = load_dataset("Salesforce/wikitext", self.wiki_config)
        dataset["val"] = dataset.pop("validation")

        dataset = dataset.filter(lambda x: x != "", input_columns=["text"])

        if self.split_functions:
            for key in self.split_functions.keys():
                dataset[key] = eval(self.split_functions[key])

        return dataset
