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
        log.FileHandler("ne_estimation.log"),
        log.StreamHandler(sys.stdout),
    ],
)
logger = log.getLogger(__name__)


def estimate_ne_for_temporal_data(
    genotype_data: GenotypeArray,
    variant_position: np.ndarray,
    metadata: pd.DataFrame,
    sample_ids: List[str],
    min_samples_per_year: int = 20,  # Minimum number of samples required to estimate Ne for a year
    max_dist: int = 50000,  # Maximum distance (bp) for LD calculation for Ne (adjust based on LD decay)
) -> pd.DataFrame:
    """
    Estimates effective population size (Ne) using Linkage Disequilibrium (LD)
    for each year present in the metadata.

    Args:
        genotype_data (allel.GenotypeArray): Filtered genotype array (NOT LD-pruned).
        variant_position (np.ndarray): Array of variant positions.
        metadata (pd.DataFrame): Sample metadata including 'Year' column.
        sample_ids (List[str]): List of sample IDs corresponding to genotype_data columns.
        min_samples_per_year (int): Minimum number of samples required in a year to
                                    attempt Ne estimation.
        max_dist (int): Maximum physical distance (bp) for considering SNP pairs
                        in LD calculation.

    Returns:
        pd.DataFrame: DataFrame with Ne estimates per year.
    """
    logger.info("Starting Ne estimation per year using LD-based method.")

    # Ensure metadata is aligned with sample_ids
    metadata_indexed = metadata.set_index("Sample")

    unique_years = sorted(metadata["Year"].unique())
    ne_results = []

    for year in unique_years:
        logger.info(f"Estimating Ne for year: {year}")

        # Select samples for the current year
        year_samples = metadata_indexed[metadata_indexed["Year"] == year].index.tolist()

        if len(year_samples) < min_samples_per_year:
            logger.warning(
                f"Skipping year {year}: Only {len(year_samples)} samples, less than required {min_samples_per_year}."
            )
            continue

        # Get indices of these samples in the overall sample_ids list
        # This requires careful alignment as sample_ids might have been filtered and reordered
        sample_indices_for_year = [
            idx for idx, sid in enumerate(sample_ids) if sid in year_samples
        ]

        if len(sample_indices_for_year) == 0:
            logger.warning(
                f"No samples found in main sample_ids list for year {year} after alignment. Skipping."
            )
            continue

        gt_year = genotype_data.take(sample_indices_for_year, axis=1)

        # Convert to n_alt (number of alt alleles)
        # For haploid Pf, this will typically be 0 or 1.
        gn_year = gt_year.to_n_alt()

        logger.info(
            f"Year {year}: Genotype array shape for Ne estimation: {gn_year.shape}"
        )

        if (
            gn_year.shape[0] < 100 or gn_year.shape[1] < min_samples_per_year
        ):  # Need enough variants and samples
            logger.warning(
                f"Skipping year {year}: Insufficient variants or samples ({gn_year.shape[0]} variants, {gn_year.shape[1]} samples)."
            )
            continue

        # Estimate Ne using allel.stats.ne_estimator
        # 'distances' parameter expects 1D array of distances between SNPs
        # 'r_squared' parameter expects 1D array of r^2 values

        # First, calculate Rogers-Huff r^2 for LD decay
        # allel.rogers_huff_r2 returns a 2D array, we need the upper triangle for unique pairs
        r_squared, ld_distances = allel.stats.rogers_huff_r_squared(
            gn_year,
            positions=variant_position,
            max_dist=max_dist,  # Consider pairs up to max_dist apart
            is_phase_known=False,  # We have genotypes, not phased haplotypes
            fill=np.nan,  # Missing values
        )

        # Filter out NaN values from r_squared and corresponding distances
        valid_indices = ~np.isnan(r_squared) & (
            ld_distances > 0
        )  # Exclude self-comparisons
        r_squared_valid = r_squared[valid_indices]
        ld_distances_valid = ld_distances[valid_indices]

        if len(r_squared_valid) == 0:
            logger.warning(
                f"Skipping year {year}: No valid LD pairs found for Ne estimation."
            )
            continue

        # Fit the Ne model
        try:
            # Ne_estimator needs r^2 values and distances.
            # You might need to adjust the alpha parameter based on your data.
            # A default alpha is 1 (recombination rate = 1 centimorgan per megabase, standard for humans)
            # For Pf, recombination rates are much higher, potentially ~6-10 cM/kb
            # Let's use a placeholder 'alpha' and note it.
            # For P. falciparum, a higher recombination rate (e.g., alpha=10, assuming distance in bp) is plausible.
            # alpha = c / (1 + c) where c is recombination rate per base pair.
            # If c = 10 cM/kb = 10 * 10^-2 M/kb = 10 * 10^-5 M/bp = 10^-4 per bp
            # Alpha for allel.ne_estimator is `C` in the formula `r^2 = 1/(1 + C*d)`.
            # C = 4 * Ne * c (recombination rate per base pair)
            # This is complex. For a first pass, let's use default or a reasonable guess.
            # If the output is bad, this is the first parameter to tune.
            # Let's use default alpha=1 for now.

            # Use `allel.stats.ne_estimator` with block_size for faster computation if data is large.
            # Or directly fit the model `y = 1 / (1 + C*x)`

            # allel.stats.ne_estimator function takes r_squared and genetic_distances.
            # We have physical_distances (ld_distances_valid).
            # We need to convert physical distance to genetic distance (if default alpha used).
            # Or use a fitting approach directly.

            # A common approach for LD-based Ne is to plot 1/r^2 vs distance
            # and fit a line. Slope is related to Ne.

            # Re-evaluating allel.stats.ne_estimator usage:
            # It expects `genetic_distances`, not `physical_distances`.
            # We need a recombination map or assume uniform recombination for physical to genetic conversion.
            # Or, we can use a simpler 1/r^2 vs. distance fit.

            # Let's use a direct curve fit for 1/r^2 = 1 + C*d where C = 4*Ne*c.
            # c is recombination rate per base pair.
            # Assuming an average c for Pf, e.g., 6.5e-9 recombinations/bp (approx. 6.5cM/Mb)
            # (Ref: Miles et al. 2016, Nature Genetics; other sources might give different estimates)

            # Filter for meaningful r_squared values (e.g., > 0.1) and distances
            # to avoid noise, but keep enough points for fit.
            # The fit typically uses the relationship 1/r^2 - 1 = 4 * Ne * c * d
            # Let y = 1/r^2 - 1, x = d. Then y = (4 * Ne * c) * x. Slope = 4 * Ne * c.

            y_data = (1 / r_squared_valid) - 1
            x_data = ld_distances_valid

            # Filter out infinite or very large values from y_data
            finite_mask = np.isfinite(y_data) & (
                y_data >= 0
            )  # r^2 should be <= 1, so 1/r^2 >= 1 -> y_data >= 0
            y_data = y_data[finite_mask]
            x_data = x_data[finite_mask]

            if len(x_data) < 10:  # Need enough points for a robust linear fit
                logger.warning(
                    f"Skipping year {year}: Insufficient valid LD pairs for linear fit."
                )
                continue

            # Perform linear regression to find the slope (m)
            # y = m*x + b
            # We are interested in m, which is 4 * Ne * c

            # Using numpy.polyfit for a simple linear fit
            coeffs, residuals, rank, singular_values, rcond = np.polyfit(
                x_data, y_data, 1, full=True
            )
            slope = coeffs[0]  # This is 4 * Ne * c

            # Assume an average recombination rate for Pf (c in recombinations/base pair)
            # A common estimate for Pf is around 6.5 cM/Mb or 6.5e-9 recombinations/bp
            # Check for current literature for more precise estimates for Nigerian Pf.
            recombination_rate_per_bp = 6.5e-9  # Example value for P. falciparum

            if slope > 0 and recombination_rate_per_bp > 0:
                estimated_ne = slope / (4 * recombination_rate_per_bp)
            else:
                estimated_ne = np.nan
                logger.warning(
                    f"Year {year}: Slope for Ne estimation was non-positive or recombination rate zero. Cannot estimate Ne."
                )

            logger.info(f"Year {year}: Estimated Ne = {estimated_ne:.2f}")
            ne_results.append(
                {
                    "Year": year,
                    "Estimated_Ne": estimated_ne,
                    "Num_Samples": gn_year.shape[1],
                    "Slope_4Nec": slope,
                    "Num_LD_Pairs": len(x_data),
                }
            )

        except Exception as e:
            logger.error(f"Error estimating Ne for year {year}: {e}")
            ne_results.append(
                {
                    "Year": year,
                    "Estimated_Ne": np.nan,
                    "Num_Samples": gn_year.shape[1],
                    "Slope_4Nec": np.nan,
                    "Num_LD_Pairs": len(x_data) if "x_data" in locals() else 0,
                }
            )

    df_ne_results = pd.DataFrame(ne_results)
    df_ne_results.sort_values(by="Year", inplace=True)
    return df_ne_results


def plot_ne_trends(df_ne_results: pd.DataFrame, output_dir: str):
    """
    Plots the estimated effective population size over time.

    Args:
        df_ne_results (pd.DataFrame): DataFrame with 'Year' and 'Estimated_Ne'.
        output_dir (str): Directory to save plots.
    """
    if df_ne_results.empty or df_ne_results["Estimated_Ne"].isnull().all():
        logger.warning("No valid Ne estimates to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out NaN Ne values for plotting
    plot_data = df_ne_results.dropna(subset=["Estimated_Ne"])

    ax.plot(
        plot_data["Year"],
        plot_data["Estimated_Ne"],
        marker="o",
        linestyle="-",
        color="purple",
        linewidth=2,
    )

    ax.set_title("Temporal Trends in Effective Population Size (Ne) in Nigeria")
    ax.set_xlabel("Year")
    ax.set_ylabel("Estimated Effective Population Size (Ne)")
    ax.set_xticks(sorted(plot_data["Year"].unique()))
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add labels for each point
    for i, row in plot_data.iterrows():
        ax.text(
            row["Year"],
            row["Estimated_Ne"],
            f"{row['Estimated_Ne']:.0f}",
            ha="right",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, "temporal_ne_trend.png")
    plt.savefig(plot_filename, dpi=300)
    logger.info(f"Saved temporal Ne trend plot to {plot_filename}")
    plt.close(fig)


def main(
    genotype_data: GenotypeArray,
    variant_position: np.ndarray,
    metadata: pd.DataFrame,
    sample_ids: List[str],
):
    """
    Main function to run Ne estimation analysis.
    """
    logger.info("--- Starting Effective Population Size (Ne) estimation analysis ---")
    output_dir = "../result/ne_estimates"

    ne_estimates_df = estimate_ne_for_temporal_data(
        genotype_data, variant_position, metadata, sample_ids
    )

    if not ne_estimates_df.empty:
        df_path = os.path.join(output_dir, "temporal_ne_estimates.csv")
        ne_estimates_df.to_csv(df_path, index=False)
        logger.info(f"Saved temporal Ne estimates to {df_path}")
        plot_ne_trends(ne_estimates_df, output_dir)
    else:
        logger.warning("No Ne estimates were calculated, skipping plotting.")

    logger.info("--- Completed Effective Population Size (Ne) estimation analysis ---")


if __name__ == "__main__":
    print("This script is intended to be called from pipeline.py")
    # Example dummy data for testing (adjust as needed)
    # import allel
    # import pandas as pd
    # import numpy as np
    # # Create dummy genotype data (1000 variants, 100 samples)
    # gt_dummy = allel.GenotypeArray(np.random.randint(0, 2, size=(1000, 100, 2)))
    # pos_dummy = np.sort(np.random.randint(1, 1000000, size=1000))
    # sample_ids_dummy = [f"Sample_{i}" for i in range(100)]
    # metadata_dummy = pd.DataFrame({
    #     'Sample': sample_ids_dummy,
    #     'Year': np.random.choice([2017, 2018, 2019, 2020], size=100),
    #     'QC pass': True,
    #     'Country': 'Nigeria'
    # })
    # # Make sure some samples are assigned to each year
    # metadata_dummy.loc[0:24, 'Year'] = 2017
    # metadata_dummy.loc[25:49, 'Year'] = 2018
    # metadata_dummy.loc[50:74, 'Year'] = 2019
    # metadata_dummy.loc[75:99, 'Year'] = 2020
    #
    # main(gt_dummy, pos_dummy, metadata_dummy, sample_ids_dummy)
