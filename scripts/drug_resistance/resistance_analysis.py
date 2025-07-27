import logging as log
import sys
import pandas as pd
import numpy as np
import allel
import matplotlib.pyplot as plt
import os

from allel import GenotypeArray
from typing import List

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        log.FileHandler("drug_resistance_analysis.log"),
        log.StreamHandler(sys.stdout),
    ],
)
logger = log.getLogger(__name__)

# Define common P. falciparum drug resistance mutations
# This dictionary can be expanded as needed.
# Format: {Gene_name: {Mutation_Name: {Chromosome: "Pf3D7_XX_v3", Position: XXX, Ref: "A", Alt: "T"}}}
# Note: For simplicity, we'll assume biallelic SNPs. If multi-allelic, more complex logic is needed.
# You will need to verify these exact positions and alleles against a reliable source
# (e.g., PlasmoDB, MalariaGEN's own DRM data if available) for the Pf8 reference.
# These are illustrative examples.
DRUG_RESISTANCE_MARKERS = {
    "pfcrt": {
        "K76T": {
            "Chromosome": "Pf3D7_07_v3",
            "Position": 403273,
            "Ref": "A",
            "Alt": "G",
        },  # Example for K76T
        # Add other pfcrt mutations like N75E, M74I, A220S, Q271E, I356T, R371I
    },
    "pfmdr1": {
        "N86Y": {
            "Chromosome": "Pf3D7_05_v3",
            "Position": 958694,
            "Ref": "A",
            "Alt": "T",
        },  # Example for N86Y
        "Y184F": {
            "Chromosome": "Pf3D7_05_v3",
            "Position": 960814,
            "Ref": "T",
            "Alt": "A",
        },  # Example for Y184F
        "D1246Y": {
            "Chromosome": "Pf3D7_05_v3",
            "Position": 961747,
            "Ref": "G",
            "Alt": "A",
        },  # Example for D1246Y
    },
    "k13": {
        "C580Y": {
            "Chromosome": "Pf3D7_13_v3",
            "Position": 1725204,
            "Ref": "T",
            "Alt": "A",
        },  # Example for C580Y
        "R539T": {
            "Chromosome": "Pf3D7_13_v3",
            "Position": 1725088,
            "Ref": "G",
            "Alt": "C",
        },  # Example for R539T
        # Add other k13 propeller mutations
    },
    "pfdhfr": {
        "N51I": {
            "Chromosome": "Pf3D7_04_v3",
            "Position": 761899,
            "Ref": "A",
            "Alt": "G",
        },  # Example for N51I
        "C59R": {
            "Chromosome": "Pf3D7_04_v3",
            "Position": 761922,
            "Ref": "C",
            "Alt": "G",
        },  # Example for C59R
        "S108N": {
            "Chromosome": "Pf3D7_04_v3",
            "Position": 762080,
            "Ref": "T",
            "Alt": "A",
        },  # Example for S108N
    },
    "pfdhps": {
        "A437G": {
            "Chromosome": "Pf3D7_08_v3",
            "Position": 530495,
            "Ref": "C",
            "Alt": "G",
        },  # Example for A437G
        "K540E": {
            "Chromosome": "Pf3D7_08_v3",
            "Position": 530825,
            "Ref": "A",
            "Alt": "C",
        },  # Example for K540E
        # Add other pfdhps mutations like S436A, A581G, A613S/T
    },
}


def get_drm_allele_frequencies(
    genotype_data: GenotypeArray,
    variant_chromosome: np.ndarray,
    variant_position: np.ndarray,
    metadata: pd.DataFrame,
    sample_ids: List[str],
    output_dir: str = "../result/drug_resistance_trends",
) -> pd.DataFrame:
    """
    Calculates the temporal allele frequencies for defined drug resistance markers.

    Args:
        genotype_data (allel.GenotypeArray): Filtered genotype array.
        variant_chromosome (np.ndarray): Array of chromosome for each variant.
        variant_position (np.ndarray): Array of position for each variant.
        metadata (pd.DataFrame): Sample metadata including 'Year' column.
        sample_ids (List[str]): List of sample IDs corresponding to genotype_data columns.
        output_dir (str): Directory to save plots.

    Returns:
        pd.DataFrame: DataFrame with temporal frequencies for each DRM.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Ensure metadata and sample_ids are aligned with genotype_data
    # This is crucial because filtering in pipeline.py might have reordered/subsetted them
    metadata_indexed = metadata.set_index("Sample")
    sample_indices = [metadata_indexed.index.get_loc(sid) for sid in sample_ids]
    aligned_metadata = metadata_indexed.iloc[sample_indices].reset_index()

    for gene, mutations in DRUG_RESISTANCE_MARKERS.items():
        for mut_name, details in mutations.items():
            chrom_target = details["Chromosome"]
            pos_target = details["Position"]
            ref_allele = details["Ref"]
            alt_allele = details["Alt"]

            # Find the index of the variant in the filtered genotype data
            # Need to handle case-insensitivity or exact string matching for chromosome names
            variant_idx = np.where(
                (variant_chromosome == chrom_target) & (variant_position == pos_target)
            )[0]

            if len(variant_idx) == 0:
                logger.warning(
                    f"DRM {gene}:{mut_name} ({chrom_target}:{pos_target}) not found in filtered data. Skipping."
                )
                continue
            elif len(variant_idx) > 1:
                logger.warning(
                    f"Multiple matches for {gene}:{mut_name}. Using the first one."
                )
                variant_idx = variant_idx[0]
            else:
                variant_idx = variant_idx[0]

            # Extract genotypes for this specific variant across all samples
            gt_variant = genotype_data[variant_idx, :]  # Shape (n_samples, ploidy)

            # Ensure the variant is biallelic and matches expected ref/alt, or handle appropriately
            # For allel.GenotypeArray, 0 is ref, 1 is alt, 2 is alt2 etc.
            # We assume it's biallelic already filtered in pipeline.py
            # If your ref/alt mapping differs, you might need a lookup

            # Count resistance allele (alt allele) for each sample
            # Assuming resistance is associated with the alternative allele (1)
            # You might need to adjust this if resistance is the ref or a specific multi-allelic variant
            allele_counts_per_sample = gt_variant.count_alleles(axis=1)[
                :, 1
            ]  # Count of alt alleles per sample

            # Map sample_ids to allele counts and years
            sample_data = pd.DataFrame(
                {
                    "Sample": sample_ids,
                    "Year": aligned_metadata["Year"].values,
                    "ResistanceAlleleCount": allele_counts_per_sample,
                    "TotalAlleles": gt_variant.to_n_haps(fill=-1).sum(
                        axis=1
                    ),  # Count of non-missing alleles
                }
            )

            # Aggregate by year
            # For diploid organisms, each sample has 2 alleles. For P. falciparum (haploid in blood stage), it's 1 allele per sample.
            # Your allel.GenotypeArray treats P. falciparum as diploid internally (n_samples, n_variants, ploidy=2)
            # count_alleles() for a single sample is (num_alleles). So for diploid it's 2, for haploid it's 1.
            # Allel's count_alleles() on the full array is (n_variants, n_alleles)
            # Let's verify how allel handles Pf haploidy/ploidy in your data.
            # Assuming gt.to_n_alt() gives 0, 1 for samples. So total alleles per sample is effectively 1 if haploid or 2 if diploid and heterozygous

            # A more robust way for frequency:
            # First, filter for non-missing genotypes for this variant
            loc_present = ~gt_variant.is_missing()

            # Get actual genotypes (0 for ref, 1 for alt for non-missing)
            genotypes_present = gt_variant.compress(loc_present, axis=0)
            sample_ids_present = np.array(sample_ids)[loc_present]
            years_present = aligned_metadata.loc[loc_present, "Year"].values

            if len(genotypes_present) == 0:
                logger.warning(
                    f"No non-missing genotypes for {gene}:{mut_name}. Skipping frequency calculation."
                )
                continue

            # Count alleles for the subset of present samples
            ac_present = allel.AlleleCountsArray(genotypes_present.count_alleles())

            # Get allele frequencies for this variant
            # If the resistance allele is the ALT allele (index 1)
            resistance_allele_freq_all_present = ac_present[:, 1] / ac_present.sum(
                axis=1
            )
            # This 'resistance_allele_freq_all_present' is a single value for this specific mutation

            # Group by year for samples that have data for this specific variant
            temp_df = pd.DataFrame(
                {
                    "Sample": sample_ids_present,
                    "Year": years_present,
                    "Allele_Type": genotypes_present.to_n_alt().flatten(),  # 0 for ref, 1 for alt (assuming haploid or 0/0, 0/1, 1/1)
                }
            )

            # Ensure 'Allele_Type' is numeric for sum/count
            temp_df["Allele_Type"] = pd.to_numeric(
                temp_df["Allele_Type"], errors="coerce"
            ).fillna(
                0
            )  # Treat missing as ref for counting

            # Calculate resistance allele count and total alleles per year
            # Sum 'Allele_Type' (0 for ref, 1 for alt) to get total alt alleles
            # Count occurrences to get total observed alleles (2 per diploid sample, 1 per haploid sample)
            # Given Pf is haploid, Allel.GenotypeArray (n_variants, n_samples, ploidy=2) means each sample has two entries.
            # If it's a haploid call, often one entry is set to missing (-1) and the other is 0 or 1.
            # A simpler way: just count alt alleles (1) vs ref alleles (0) in the to_n_alt() representation.

            # Let's simplify and assume 0 = reference, 1 = alternate allele for the resistance allele
            # Count the number of times the alt allele (1) appears for this variant in samples for each year
            freq_by_year = (
                temp_df.groupby("Year")["Allele_Type"]
                .agg(
                    resistance_allele_count="sum",
                    total_observed_alleles=lambda x: (
                        x >= 0
                    ).sum(),  # Count non-missing (0 or 1)
                )
                .reset_index()
            )

            # Calculate frequency
            freq_by_year["Frequency"] = (
                freq_by_year["resistance_allele_count"]
                / freq_by_year["total_observed_alleles"]
            )

            # Add metadata for the result
            freq_by_year["Gene"] = gene
            freq_by_year["Mutation"] = mut_name
            freq_by_year["Chromosome"] = chrom_target
            freq_by_year["Position"] = pos_target

            results.append(freq_by_year)
            logger.info(f"Calculated temporal frequencies for {gene}:{mut_name}.")

    if not results:
        logger.warning("No drug resistance markers were found or processed.")
        return pd.DataFrame()

    all_drm_frequencies = pd.concat(results, ignore_index=True)
    all_drm_frequencies.sort_values(by=["Gene", "Mutation", "Year"], inplace=True)

    # Plotting
    plot_temporal_allele_frequencies(all_drm_frequencies, output_dir)
    return all_drm_frequencies


def plot_temporal_allele_frequencies(df: pd.DataFrame, output_dir: str):
    """
    Plots the temporal allele frequencies for each drug resistance marker.

    Args:
        df (pd.DataFrame): DataFrame containing 'Year', 'Frequency', 'Gene', 'Mutation'.
        output_dir (str): Directory to save plots.
    """
    if df.empty:
        logger.warning("No data to plot for drug resistance trends.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # Get unique genes for separate plots if desired, or plot all on one
    unique_genes = df["Gene"].unique()

    for gene in unique_genes:
        gene_df = df[df["Gene"] == gene]

        if gene_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        # Group by mutation to plot each mutation's trend
        for mutation_name, mut_data in gene_df.groupby("Mutation"):
            ax.plot(
                mut_data["Year"],
                mut_data["Frequency"],
                marker="o",
                linestyle="-",
                label=mutation_name,
            )

        ax.set_title(f"Temporal Allele Frequencies for {gene} Mutations in Nigeria")
        ax.set_xlabel("Year")
        ax.set_ylabel("Resistance Allele Frequency")
        ax.set_ylim(0, 1)  # Frequencies are between 0 and 1
        ax.set_xticks(sorted(gene_df["Year"].unique()))
        ax.grid(True)
        ax.legend(title="Mutation")
        plt.tight_layout()

        plot_filename = os.path.join(
            output_dir, f"{gene}_temporal_allele_frequencies.png"
        )
        plt.savefig(plot_filename, dpi=300)
        logger.info(
            f"Saved temporal allele frequency plot for {gene} to {plot_filename}"
        )
        plt.close(fig)

    # Optionally, a single plot with all mutations (might be messy if too many)
    fig_all, ax_all = plt.subplots(figsize=(15, 8))
    for (gene, mutation_name), mut_data in df.groupby(["Gene", "Mutation"]):
        ax_all.plot(
            mut_data["Year"],
            mut_data["Frequency"],
            marker="o",
            linestyle="-",
            label=f"{gene}: {mutation_name}",
        )
    ax_all.set_title(
        "Temporal Allele Frequencies for Key Drug Resistance Markers in Nigeria"
    )
    ax_all.set_xlabel("Year")
    ax_all.set_ylabel("Resistance Allele Frequency")
    ax_all.set_ylim(0, 1)
    ax_all.set_xticks(sorted(df["Year"].unique()))
    ax_all.grid(True)
    ax_all.legend(title="Marker", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plot_filename_all = os.path.join(
        output_dir, "all_drm_temporal_allele_frequencies.png"
    )
    plt.savefig(plot_filename_all, dpi=300)
    logger.info(f"Saved combined temporal allele frequency plot to {plot_filename_all}")
    plt.close(fig_all)


def main(
    genotype_data: GenotypeArray,
    variant_chromosome: np.ndarray,
    variant_position: np.ndarray,
    metadata: pd.DataFrame,
    sample_ids: List[str],
):
    """
    Main function to run drug resistance analysis.
    """
    logger.info("Starting temporal drug resistance allele frequency analysis.")
    drm_frequencies_df = get_drm_allele_frequencies(
        genotype_data, variant_chromosome, variant_position, metadata, sample_ids
    )
    if not drm_frequencies_df.empty:
        df_path = "../result/drug_resistance_trends/temporal_drm_frequencies.csv"
        drm_frequencies_df.to_csv(df_path, index=False)
        logger.info(f"Saved temporal DRM frequencies to {df_path}")
    logger.info("Completed temporal drug resistance allele frequency analysis.")


if __name__ == "__main__":
    print("This script is intended to be called from pipeline.py")
