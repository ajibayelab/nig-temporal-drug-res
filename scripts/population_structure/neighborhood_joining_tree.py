#!/usr/bin/env python3
"""
Neighborhood Joining Tree Analysis for Population Structure

This module performs phylogenetic analysis of population genetic data using the
Neighbor-Joining (NJ) algorithm. It processes genomic variant data stored in zarr
format to construct evolutionary trees that reveal population structure and
relationships between samples or populations.

The analysis pipeline includes:
1. Loading and preprocessing genomic data from zarr arrays
2. Computing genetic distance matrices between samples/populations
3. Constructing neighbor-joining trees
4. Visualizing and exporting results

Key Features:
- Efficient handling of large genomic datasets via zarr
- Multiple genetic distance metrics (Hamming, p-distance, etc.)
- Population-level aggregation and analysis
- Tree visualization and export capabilities
- Memory-efficient processing for large datasets

Dependencies:
- zarr: For efficient array storage and access
- numpy: Numerical computations and array operations
- scipy: Distance calculations and clustering
- matplotlib/seaborn: Visualization
- Bio.Phylo: Phylogenetic tree handling (optional)
- pandas: Data manipulation and sample metadata

Example Usage:
    analyzer = NJTreeAnalyzer('genomic_data.zarr')
    analyzer.load_data()
    analyzer.compute_distances(metric='hamming')
    tree = analyzer.build_nj_tree()
    analyzer.plot_tree(tree, output='population_tree.png')

Author: Isong Josiah
Date: 2025-01-10
Version: 1.0
"""

import logging as log
import sys
from numpy import mean
from numpy.random import sample
import pandas as pd
import xarray
from typing import List


# Configure logging
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        log.FileHandler("malariagen_extraction.log"),
        log.StreamHandler(sys.stdout),
    ],
)
logger = log.getLogger(__name__)


def main(
    pf8_metadata_df: pd.DataFrame, variant_data: xarray.Dataset, sample_ids: List[str]
):
    import anjl
    import s3fs
    import numpy as np
    from skbio.stats.ordination import pcoa
    import os

    # retrieve data
    logger.info("--- Running Population Structure Analysis ---")

    # NOTE:check if I already have the subsetted distance matrix file
    # if I don't have it, pull and subset otherwise use the saved file
    # If you are using a different subset from
    #   - "Country" == "Nigeria"
    #   - "QC pass" == True
    # then you should fetch the original and subset
    mean_distance_matrix = None
    qc_matrix = None
    qc_dist_matrix = "../data/qc_matrix.npy"
    if not os.path.exists(qc_dist_matrix):
        # NOTE: use the pf8 genotype metadata from data if it
        # exists
        dist_mat_path = "../data/Pf8_mean_genotype_distance.npy"
        if not os.path.exists(dist_mat_path):
            logger.info("dist_mat_path does not exist")
            logger.info("Loading distance matrix from pf8")
            dist_mat_path = "s3://pf8-release/Pf8_mean_genotype_distance.npy"
            s3_config = {
                "signature_version": "s3",
                "s3": {"addressing_style": "virtual"},
            }
            fs = s3fs.S3FileSystem(
                anon=True,
                endpoint_url="https://cog.sanger.ac.uk",
                config_kwargs=s3_config,
            )
            with fs.open(dist_mat_path, "rb") as f:
                mean_distance_matrix = np.load(f)

            logger.info("--- Loaded Pf8 genotype distance matrix ---")
        else:
            logger.info("dist_mat_path exist")
            logger.info("--- loading distance matrix from fs ---")
            mean_distance_matrix = np.load(dist_mat_path)
            qc_matrix = mean_distance_matrix[sample_ids][:, sample_ids]
            np.fill_diagonal(qc_matrix, 0)
            np.save(qc_dist_matrix, qc_matrix)
            logger.info("--- saved qc distance matrix to fs ---")

    else:
        logger.info("Loading distance matrix from fs")
        qc_matrix = np.load(qc_dist_matrix)
        logger.info("Loaded distance matrix from fs")

    # perform PCoA
    logger.info("--- Computing Principal Coordinate Analysis ---")
    pcoa_results = pcoa(qc_matrix)

    # creating the tree object
    colour_dict = {
        "2020.0": "#9ecae1",
        "2017.0": "#fecc5c",
        "2019.0": "#984ea3",
        "2014.0": "#bb8129",
        "2018.0": "#dfc0eb",
        "2008.0": "#4daf4a",
        "2012.0": "#fd8d3c",
        "2011.0": "#e31a1c",
    }

    import numpy as np
    import anjl._plot as anjl_plot

    def patched_paint_internal(Z, leaf_color_values):
        n = Z.shape[0] + 1
        internal_color_values = np.full(Z.shape[0], np.nan)

        for z in range(Z.shape[0]):
            left = int(Z[z, 0])
            right = int(Z[z, 1])

            left_color = (
                leaf_color_values[left] if left < n else internal_color_values[left - n]
            )
            right_color = (
                leaf_color_values[right]
                if right < n
                else internal_color_values[right - n]
            )

            # Only assign if both sides match and are valid
            if (
                left_color == right_color
                and left_color is not None
                and left_color != ""
                and not (isinstance(left_color, float) and np.isnan(left_color))
            ):
                internal_color_values[z] = left_color

        return internal_color_values

    # monkey-patch the buggy function
    anjl_plot.paint_internal = patched_paint_internal

    if qc_matrix is None:
        logger.error("genotype distance matrix is invalid")
        return

    Z = anjl.dynamic_nj(qc_matrix, disallow_negative_distances=True)
    # ignore branch length
    Z[:, 2] = 1.0
    fig = anjl.plot(
        Z,
        leaf_data=pf8_metadata_df,
        color="Year",
        color_discrete_map=colour_dict,
        hover_name="Sample",
        hover_data=["Admin level 1"],
        line_width=3,
        marker_size=2,
        width=1500,
        height=1500,
        distance_sort=False,
        count_sort=True,
    )
    fig.write_image("../result/population_structure/pf8_nigeria_njt.png", scale=2)
    # fig.show()
    logger.info("--- completed neighborhood joining tree")


# if __name__ == "__main__":
#    main()
