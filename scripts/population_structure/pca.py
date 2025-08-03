import logging as log
import sys
from allel import GenotypeArray, pca
from allel.model.ndarray import genotype_array_count_alleles_subpop
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from pandas.core.frame import Axes
from xarray import DataTree
from zarr import meta

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        log.FileHandler("pcoa.log"),
        log.StreamHandler(sys.stdout),
    ],
)
logger = log.getLogger(__name__)


# BUG:getting ValueError: arrayy must not contain infs or NaNs
# falling back to manual PCA computation
def calculate_pca(
    metadata: pd.DataFrame,
    genotype_data: GenotypeArray,
    sample_ids: List[str],
    n_components=2,
):
    logger.info("----- Starting PCA computation -----")

    ac = genotype_data.count_alleles()
    mask = ac.allelism() > 1
    print(f"number of samples {np.sum(mask)}")
    genotype_data = genotype_data[mask]
    gn = genotype_data.to_n_alt()

    if np.any(np.isinf(gn)) or np.any(np.isnan(gn)):
        logger.error(
            "The genotype array still contains Inf or NaN values after filtering."
        )
        gn = np.nan_to_num(gn)
        logger.warning(
            "Replaced remaining Inf/NaN values with 0. Proceeding with caution."
        )

    coordinates, model = pca(gn)

    logger.info("----- PCA computed -----")

    logger.info(f"explaind model variance {model.explained_variance_}")
    logger.info(f"explaind model variance ration {model.explained_variance_ratio_}")
    logger.info(f"coordinates are {coordinates}")


def assign_clusters_by_cutoff(coordinates: np.ndarray) -> np.ndarray:
    """
    Assigns cluster IDs to samples based on simple PC coordinate cutoffs.

    Args:
        coordinates (np.ndarray): A 2D array of PCA coordinates, where
                                  the first column is PC1 and the second is PC2.

    Returns:
        np.ndarray: A 1D array of cluster IDs for each sample.
    """

    cluster_ids = np.ones(coordinates.shape[0], dtype=int)

    # NOTE: cutoffs array is in order [left PC1, right PC1, bottom PC2, top PC2]
    cutofs: Dict[int, List[int]] = {
        1: [-20, 20, -20, 20],  # main population
        2: [-20, 0, -30, -20],  # bottom left outlier
        3: [-20, 0, 80, 120],  # top left outlier
        4: [80, 120, -20, 20],  # right outlier
    }

    pc1 = coordinates[:, 0]
    pc2 = coordinates[:, 1]
    for id, values in cutofs.items():
        cluster_mask = (
            (pc1 >= values[0])
            & (pc1 <= values[1])
            & (pc2 >= values[2])
            & (pc2 <= values[3])
        )
        cluster_ids[cluster_mask] = id

    return cluster_ids


def plot_pca(metadata: pd.DataFrame, explained_variance_ratio):
    """
    Plot PCA results for P. falciparum samples with color coding by year.

    Args:
        metadata (pd.DataFrame): DataFrame containing PCA coordinates and metadata
                                Must include columns: 'PC1', 'PC2', 'Year', 'Cluster IDs'
        explained_variance_ratio (array-like): Array of explained variance ratios
                                            for each principal component
    """

    import matplotlib.pyplot as plt

    pc1_var = explained_variance_ratio[0] * 100
    pc2_var = explained_variance_ratio[1] * 100

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        metadata["PC1"],
        metadata["PC2"],
        c=metadata["Year"],
        cmap="viridis",
        alpha=0.8,
        s=70,
    )

    ax.set_xlabel(f"PC1 ({pc1_var:.2f}% variance explained)")
    ax.set_ylabel(f"PC2 ({pc2_var:.2f}% variance explained)")
    ax.set_title("PCA of P. falciparum Samples, Colored by Time")

    # Add a legend for time points if you were using discrete colors
    # For a continuous colormap, a color bar is better.
    # If you used discrete categories for colors, you would do this:
    for tp in sorted(metadata["Year"].unique()):
        ax.scatter(
            metadata.loc[metadata["Year"] == tp, "PC1"],
            metadata.loc[metadata["Year"] == tp, "PC2"],
            label=f"Year {tp}",
        )
    ax.legend(title="Year")

    plt.show()


def main(
    metadata: pd.DataFrame,
    genotype_data: GenotypeArray,
    sample_ids: List[str],
    n_components=2,
) -> Dict[int, Tuple[GenotypeArray, pd.DataFrame]]:
    import numpy as np
    from sklearn.decomposition import PCA
    from .fst import main as fst_main

    logger.info("--- Computing PCA ---")

    # NOTE: at this point we have filtered our genotype to limit
    # it to biallelic variants with statistically significant
    # alternate alleles
    gn = genotype_data.to_n_alt()
    gn_t = gn.T  # samples are now rows and variants are column
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(gn_t)
    explained_variance_ratio = pca.explained_variance_ratio_
    logger.info(
        f"Explained variance ratio of the first {n_components} PCs {explained_variance_ratio}"
    )

    # Merge PCA coordinates with metadata
    pc_columns = np.array([f"PC{i+1}" for i in range(n_components)])
    pca_df = pd.DataFrame(coords, columns=pc_columns)
    pca_df["Sample"] = sample_ids

    metadata = metadata.merge(pca_df, on="Sample", how="inner")
    cluster_ids = assign_clusters_by_cutoff(coordinates=coords)
    metadata["Cluster IDs"] = cluster_ids

    plot_pca(metadata, explained_variance_ratio)

    cluster_details: Dict[int, Tuple[GenotypeArray, pd.DataFrame]] = {}
    unique_clusters = metadata["Cluster IDs"].unique()
    for cluster_id in unique_clusters:
        cluster_mask = metadata["Cluster IDs"] == cluster_id
        genotype_subset = genotype_data.compress(cluster_mask, axis=1)
        metadata_subset: pd.DataFrame = metadata.loc[
            metadata["Cluster IDs"] == cluster_id
        ].reset_index(drop=True)
        cluster_details[cluster_id] = (
            genotype_subset,
            metadata_subset,
        )

    return cluster_details
