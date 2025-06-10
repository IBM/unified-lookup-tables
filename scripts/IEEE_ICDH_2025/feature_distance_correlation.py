import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
from Levenshtein import ratio
from scipy.stats import pearsonr
from scipy.stats._result_classes import PearsonRResult
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from m2fm.data.loaders.multiomics.tcga_augmented import MultiOmicsDataLoader
from m2fm.data.preprocessing.unicode.transforms import (
    UnicodeSeriesCompansionTransform,
    UnicodeSeriesCompansionTransformParameters,
)

unicode_key_transform_map = {
    "series": {
        "config": UnicodeSeriesCompansionTransformParameters,
        "transform": UnicodeSeriesCompansionTransform,
    },
}


@click.command()
@click.option(
    "--data_dir",
    type=click.Path(path_type=Path, exists=True),
    help="Path to directory containing data on which lut was built.",
)
@click.option(
    "--lut_dir",
    type=click.Path(path_type=Path, exists=True),
    help="Path to directory containing all luts for hyperparam search.",
)
@click.option(
    "--result_dir",
    type=click.Path(path_type=Path, exists=True),
    help="Path to save similarity correlation results.",
)
def compare_feature_string_similarity(
    data_dir: Path, lut_dir: Path, result_dir: Path
) -> PearsonRResult:
    data_loader = MultiOmicsDataLoader(data_dir, subsample=True, subsample_fraction=0.25)

    dataset = data_loader.load_splits()

    synthetic_subset = [
        dataset["test"]
        .filter(lambda example: example["target"] == i)
        .take(50)
        .select_columns(["array", "target"])
        for i in range(4)
    ]

    synthetic_subset_array = np.vstack([subset["array"] for subset in synthetic_subset])

    # # compute cosine similarity in feature space
    # cosine_matrix = cosine_similarity(synthetic_subset_array)

    # compute euclidean distance
    euclidean_matrix = euclidean_distances(synthetic_subset_array)

    # get unicode strings
    hparam_result_df = pd.DataFrame(
        columns=[
            "transform",
            "patch_size",
            "companding_max",
            "mu",
            "pearson_corr",
            "pearson_pval",
        ]
    )
    for i, lut_path in enumerate(lut_dir.glob("lut-rna-series*")):
        transform_key = lut_path.name.split("-")[2]
        param_string = re.search("p\d+cm\d+mu\d+", lut_path.name).group(0)
        patch = int(re.search("p(\d+)", param_string).group(1))
        comp_max = float(".".join(re.search("cm(\d+)", param_string).group(1)))
        mu_val = int(re.search("mu(\d+)", param_string).group(1))
        transform_params = unicode_key_transform_map[transform_key]["config"](
            patch_size=patch,
            occurrences_path=lut_path,
            companding_max=comp_max,
            mu=mu_val,
        )

        unicode_transformer = unicode_key_transform_map[transform_key]["transform"](
            transform_params
        )

        unicode_strings_list = [
            unicode_transformer.encode(sample) for sample in synthetic_subset_array
        ]

        # compute levenshtein ratio in string space

        string_matrix = pairwise_distances(unicode_strings_list, metric=ratio)

        # compute pearson correlation

        pearson_corr = pearsonr(
            np.triu(euclidean_matrix).ravel(),
            np.triu(string_matrix).ravel(),
            alternative="greater",
        )

        hparam_result_df.loc[i, "transform"] = transform_key
        hparam_result_df.loc[i, "patch_size"] = patch
        hparam_result_df.loc[i, "companding_max"] = comp_max
        hparam_result_df.loc[i, "mu"] = mu_val
        hparam_result_df.loc[i, "pearson_corr"] = pearson_corr.statistic
        hparam_result_df.loc[i, "pearson_pval"] = pearson_corr.pvalue

    hparam_result_df.to_csv(result_dir / "hparam_similarity_scores.csv")


if __name__ == "__main__":
    compare_feature_string_similarity()
