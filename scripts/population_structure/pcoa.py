import logging as log
import sys
from allel import GenotypeArray
import pandas as pd
from typing import Dict, List

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        log.FileHandler("pcoa.log"),
        log.StreamHandler(sys.stdout),
    ],
)
logger = log.getLogger(__name__)


def main(
    metadata: pd.DataFrame,
    genotype_data: GenotypeArray,
    sample_ids: List[str],
    n_components=3,
) -> GenotypeArray:
    logger.info("Computing pcoa")
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from .fst import main as fst_main

    logger.info(f"Genotype array shape: {genotype_data.shape}")
    # NOTE: at this point we have filtered our genotype to limit
    # it to biallelic variants with statistically significant
    # alternate alleles
    gn = genotype_data.to_n_alt()
    logger.info(f"The shape of n_alt is {gn.shape}")

    # samples are now rows and variants are column
    gn_t = gn.T
    # run PCA
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(gn_t)
    explained_variance_ratio = pca.explained_variance_ratio_

    logger.info(
        f"Explained variance ratio of the first {n_components} PCs {explained_variance_ratio}"
    )

    # Merge PCA coordinates with metadata
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(coords, columns=pc_columns)
    pca_df["Sample"] = sample_ids

    logger.info("Merging PCA results with metadata")
    metadata = metadata.merge(pca_df, on="Sample", how="inner")

    logger.info(f"Merged data shape: {metadata.shape}")
    logger.info(f"Samples in final analysis: {len(metadata)}")

    # plot_pca(metadata, explained_variance_ratio)

    # run Fst between the two clusters of interest
    cluster_threshold: Dict[str, List] = {
        "Cluster1": [[50, 150], [-50, 50]],  # bottom righ cluster
        "Cluster2": [[-50, 25], [100, 250]],  # top left cluster
    }
    # Define the conditions as boolean Series
    # Note the correct logical AND operator `&` for Pandas Series
    conditions = [
        # Condition for Cluster 1
        (
            (
                (metadata["PC1"] >= cluster_threshold["Cluster1"][0][0])
                & (metadata["PC1"] <= cluster_threshold["Cluster1"][0][1])
            )
            & (
                (metadata["PC2"] >= cluster_threshold["Cluster1"][1][0])
                & (metadata["PC2"] <= cluster_threshold["Cluster1"][1][1])
            )
        ),
        # Condition for Cluster 2
        (
            (
                (metadata["PC1"] >= cluster_threshold["Cluster2"][0][0])
                & (metadata["PC1"] <= cluster_threshold["Cluster2"][0][1])
            )
            & (
                (metadata["PC2"] >= cluster_threshold["Cluster2"][1][0])
                & (metadata["PC2"] <= cluster_threshold["Cluster2"][1][1])
            )
        ),
    ]

    # Define the corresponding cluster values
    choices = [
        "Cluster1",
        "Cluster2",
    ]

    # Use np.select() to choose from the choices based on the conditions
    metadata["Cluster"] = np.select(conditions, choices, default="Indistinct Cluster")

    cluster_1_index = metadata[metadata["Cluster"] == "Cluster1"].index
    cluster_2_index = metadata[metadata["Cluster"] == "Cluster2"].index
    cluster_3_index = metadata[metadata["Cluster"] == "Indistinct Cluster"].index
    print("cluster 3 index", cluster_3_index)

    fst_for_distinct_clusters = fst_main(
        genotype_data, [cluster_1_index.values, cluster_2_index.values]
    )
    fst_for_all_clusters = fst_main(
        genotype_data,
        [cluster_1_index.values, cluster_2_index.values, cluster_3_index.values],
    )

    logger.info(f"Fst for the two distinct clusters are {fst_for_distinct_clusters}")
    logger.info(f"Fst for all clusters {fst_for_all_clusters}")

    idx_1 = cluster_1_index.values
    idx_2 = cluster_2_index.values
    idx = np.concatenate((np.array(idx_1), np.array(idx_2)))
    genotype_data = genotype_data.take(idx, axis=1)
    print("genotype data shape", genotype_data.shape)
    return genotype_data


def plot_pca(metadata: pd.DataFrame, explained_variance_ratio):
    import matplotlib.pyplot as plt
    import numpy as np

    # Let's plot PC1 vs PC2
    pc1_var = explained_variance_ratio[0] * 100
    pc2_var = explained_variance_ratio[1] * 100

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a colormap to represent the time points
    # `c` parameter is used for the color, and `cmap` for the color map
    scatter = ax.scatter(
        metadata["PC1"],
        metadata["PC2"],
        c=metadata["Year"],
        cmap="viridis",  # 'viridis' is a good sequential colormap
        alpha=0.8,
        s=70,  # size of the points
    )

    # Add a color bar to show the mapping from color to time
    cbar = fig.colorbar(scatter, ax=ax, label="Time Point (Year)")
    cbar.set_ticks(np.unique(metadata["Year"]))

    # Add labels with explained variance
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
    ax.legend(title="Time Point")

    plt.show()
