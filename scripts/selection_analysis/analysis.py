import logging as log
import sys
import pandas as pd
import numpy as np
import allel
import matplotlib.pyplot as plt
import os

from allel import GenotypeArray
from typing import List, Tuple

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        log.FileHandler("selection_analysis.log"),
        log.StreamHandler(sys.stdout),
    ],
)
logger = log.getLogger(__name__)


def calculate_selection_statistics_windows(
    genotype_data: GenotypeArray,
    variant_chromosome: np.ndarray,
    variant_position: np.ndarray,
    window_size: int = 20000,  # 20 kb window
    window_step: int = 10000,  # 10 kb step
) -> pd.DataFrame:
    """
    Calculates Tajima's D and Nucleotide Diversity in sliding windows across the genome.

    Args:
        genotype_data (allel.GenotypeArray): Filtered genotype array.
        variant_chromosome (np.ndarray): Array of chromosome for each variant.
        variant_position (np.ndarray): Array of position for each variant.
        window_size (int): Size of the sliding window in base pairs.
        window_step (int): Step size for the sliding window in base pairs.

    Returns:
        pd.DataFrame: DataFrame containing window-based selection statistics.
    """
    logger.info(
        f"Calculating selection statistics in {window_size/1000}kb windows with {window_step/1000}kb step."
    )

    results = []

    unique_chroms = np.unique(variant_chromosome)
    for chrom in sorted(
        unique_chroms
    ):  # Sort chromosomes for consistent plotting order
        logger.info(f"Processing chromosome: {chrom}")

        # Select variants for the current chromosome
        loc_chrom = variant_chromosome == chrom
        gt_chrom = genotype_data.compress(loc_chrom, axis=0)
        pos_chrom = variant_position.compress(loc_chrom)

        if gt_chrom.n_variants < 2:  # Need at least 2 variants for window calculations
            logger.warning(
                f"Skipping chromosome {chrom}: Too few variants ({gt_chrom.n_variants})."
            )
            continue

        # Convert to allele counts for Tajima's D and diversity
        ac_chrom = gt_chrom.count_alleles()

        # Define windows
        # Use allel.window.moving_windows for convenience
        windows = allel.window.moving_windows(
            pos_chrom, size=window_size, step=window_step
        )

        for i, window in enumerate(windows):
            window_start = pos_chrom[window.start]
            window_end = pos_chrom[window.stop - 1]  # Last variant in window
            window_mid = (window_start + window_end) // 2

            # Ensure window is valid (e.g., has enough variants)
            if window.stop - window.start < 2:
                continue

            gt_window = gt_chrom[window.start : window.stop]
            ac_window = ac_chrom[window.start : window.stop]

            # Calculate Tajima's D
            # Handles NaN for monomorphic windows
            tajima_d = allel.stats.tajima_d(ac_window)

            # Calculate Nucleotide Diversity (pi)
            # For haploid Pf, pi = count_differences / total_pairs
            # allel.sequence_diversity calculates pi for diploid. For haploid, it's simpler:
            # pi = num_pairwise_differences / num_pairs. Or, use count_alleles_gc() for G-C content and then calc pi.
            # Using allel.sequence_diversity on the genotype array should work as it's designed for genetic data.
            nuc_diversity = allel.stats.sequence_diversity(
                pos_chrom[window.start : window.stop], ac_window
            )

            # Note: For Pf (often haploid at blood stage), it might be more accurate to calculate from haplotype array
            # or ensure sequence_diversity is interpreted correctly for effective haploidy.
            # For `allel.stats.sequence_diversity(pos, ac)`, if `ac` are allele counts from a haploid population
            # (e.g., sum of ac for a variant is N_samples), it should be fine.

            results.append(
                {
                    "Chromosome": chrom,
                    "Window_Start": window_start,
                    "Window_End": window_end,
                    "Window_Mid": window_mid,
                    "Tajima_D": tajima_d,
                    "Nucleotide_Diversity": nuc_diversity,
                    "Num_Variants_in_Window": gt_window.n_variants,
                }
            )

    df_results = pd.DataFrame(results)
    logger.info(
        f"Finished calculating selection statistics. Total windows: {len(df_results)}"
    )
    return df_results


def plot_selection_statistics(df_results: pd.DataFrame, output_dir: str):
    """
    Plots Tajima's D and Nucleotide Diversity across chromosomes.

    Args:
        df_results (pd.DataFrame): DataFrame containing window-based selection statistics.
        output_dir (str): Directory to save plots.
    """
    if df_results.empty:
        logger.warning("No selection statistics data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Tajima's D plot
    fig_td, ax_td = plt.subplots(figsize=(15, 7))
    allel.plot.fig_by_chrom(
        df_results,
        "Window_Mid",
        "Tajima_D",
        "Chromosome",
        ax=ax_td,
        color="blue",
        s=10,
        label="Tajima's D",
    )
    ax_td.axhline(0, color="grey", linestyle="--", lw=0.8)  # Neutral expectation
    ax_td.set_ylabel("Tajima's D")
    ax_td.set_title("Tajima's D across P. falciparum Chromosomes in Nigeria")
    ax_td.legend()
    plt.tight_layout()
    plot_filename_td = os.path.join(output_dir, "tajima_d_plot.png")
    plt.savefig(plot_filename_td, dpi=300)
    logger.info(f"Saved Tajima's D plot to {plot_filename_td}")
    plt.close(fig_td)

    # Nucleotide Diversity plot
    fig_pi, ax_pi = plt.subplots(figsize=(15, 7))
    allel.plot.fig_by_chrom(
        df_results,
        "Window_Mid",
        "Nucleotide_Diversity",
        "Chromosome",
        ax=ax_pi,
        color="green",
        s=10,
        label="Nucleotide Diversity (π)",
    )
    ax_pi.set_ylabel("Nucleotide Diversity (π)")
    ax_pi.set_title("Nucleotide Diversity across P. falciparum Chromosomes in Nigeria")
    ax_pi.legend()
    plt.tight_layout()
    plot_filename_pi = os.path.join(output_dir, "nucleotide_diversity_plot.png")
    plt.savefig(plot_filename_pi, dpi=300)
    logger.info(f"Saved Nucleotide Diversity plot to {plot_filename_pi}")
    plt.close(fig_pi)


def main(
    genotype_data: GenotypeArray,
    variant_chromosome: np.ndarray,
    variant_position: np.ndarray,
):
    """
    Main function to run selective sweep analysis.
    """
    logger.info("Starting selective sweep analysis.")
    output_dir = "../result/selective_sweeps"

    # It's important to use the genotype data *after* filtering for missingness, MAC, and MAF
    # but *before* LD pruning, as LD pruning might remove variants critical for accurate window calculations.

    selection_stats_df = calculate_selection_statistics_windows(
        genotype_data, variant_chromosome, variant_position
    )

    if not selection_stats_df.empty:
        df_path = os.path.join(output_dir, "selection_statistics.csv")
        selection_stats_df.to_csv(df_path, index=False)
        logger.info(f"Saved selection statistics to {df_path}")
        plot_selection_statistics(selection_stats_df, output_dir)
    else:
        logger.warning("No selection statistics were calculated, skipping plotting.")

    logger.info("Completed selective sweep analysis.")


if __name__ == "__main__":
    print("This script is intended to be called from pipeline.py")
    # You could add dummy data for direct testing similar to drug_resistance_analysis.py
