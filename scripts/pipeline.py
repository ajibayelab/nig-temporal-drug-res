import logging as log
from re import T
import sys
import os
from typing import Tuple, List

import allel
from numpy.random import sample
import pandas
from xarray import Dataset
import numpy as np

import retrieve_data

import population_structure.pca as pca
import population_structure.linkage_disequilibrum as ld
import drug_resistance.resistance_analysis as dra
import selection_analysis.analysis as sa
import population_structure.ne_estimation as nea

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


def retrieve_genotype(
    snp_dataset: Dataset,
) -> Tuple[allel.GenotypeArray, np.ndarray, np.ndarray]:
    import os

    logger.info("--- Retrieving genotype information ---")
    genotype_path = "../data/pf8_nigerian_genotype.npy"
    chrom_path = "../data/pf8_nigeria_variant_chromosome.npy"
    pos_path = "../data/pf8_nigeria_variant_position.npy"

    biallelic_genotype_path = "../data/pf8_nigerian_biallelic_genotype.npy"
    biallelic_chrom_path = "../data/pf8_nigeria_biallelic_variant_chromosome.npy"
    biallelic_pos_path = "../data/pf8_nigeria_biallelic_variant_position.npy"

    gt: allel.GenotypeArray
    chrom_data: np.ndarray
    pos_data: np.ndarray

    # If we have all three biallelic filtered files. Use them
    # and return
    if (
        os.path.exists(biallelic_chrom_path)
        and os.path.exists(biallelic_genotype_path)
        and os.path.exists(biallelic_pos_path)
    ):
        logger.info("Parsing saved biallelic variants")
        snp_data = np.load(biallelic_genotype_path)
        gt = allel.GenotypeArray(snp_data)

        chrom_data = np.load(biallelic_chrom_path, allow_pickle=True)
        pos_data = np.load(biallelic_pos_path, allow_pickle=True)

        return gt, chrom_data, pos_data

    # if we don't have the genotype detail yet. Fetch it
    if not os.path.exists(genotype_path):
        snp_data = snp_dataset.call_genotype
        gt = allel.GenotypeArray(snp_data)
        logger.info("--- Saving genotype to file ---")
        np.save(genotype_path, gt)
        logger.info("--- Saved genotype to file ---")
    else:
        snp_data = np.load(genotype_path)
        gt = allel.GenotypeArray(snp_data)

    # if we don't have the variant chromosome yet. Fetch it
    if not os.path.exists(chrom_path):
        chrom_data = snp_dataset["variant_chrom"].values
        logger.info("--- Saving variant chromosome information to file---")
        np.save(chrom_path, chrom_data)
        logger.info("--- Saved  variant chromosome to file ---")
    else:
        chrom_data = np.load(chrom_path, allow_pickle=True)

    # if we don't have the variant position yet. Fetch it
    if not os.path.exists(pos_path):
        pos_data = snp_dataset["variant_position"].values
        logger.info("--- Saving variant position information to file ---")
        np.save(pos_path, pos_data)
        logger.info("--- Saved variant position to file")
    else:
        pos_data = np.load(pos_path, allow_pickle=True)

    # filter out non variants
    ac = gt.count_alleles()
    variant_filter = ac.is_variant()
    gt = gt[variant_filter]
    chrom_data = chrom_data[variant_filter]
    pos_data = pos_data[variant_filter]

    # filter out non biallelic variants
    ac = gt.count_alleles()
    biallelic_filter = ac.is_biallelic()
    gt = gt[biallelic_filter]
    chrom_data = chrom_data[biallelic_filter]
    pos_data = pos_data[biallelic_filter]

    # Save biallelic variants
    biallelic_genotype_path = "../data/pf8_nigerian_biallelic_genotype.npy"
    biallelic_chrom_path = "../data/pf8_nigeria_biallelic_variant_chromosome.npy"
    biallelic_pos_path = "../data/pf8_nigeria_biallelic_variant_position.npy"

    logger.info("--- Saving bialelic genotypes ---")
    np.save(biallelic_genotype_path, gt)
    logger.info("--- Saved bialelic genotypes ---")

    logger.info("--- Saving bialelic variant chromosome ---")
    np.save(biallelic_chrom_path, chrom_data)
    logger.info("--- Saved bialelic variant chromosome ---")

    logger.info("--- Saving bialelic variant position ---")
    np.save(biallelic_pos_path, pos_data)
    logger.info("--- Saved bialelic variant position ---")

    return gt, chrom_data, pos_data


def filter_missing_genotype(
    gt: allel.GenotypeArray,
    variant_chromosome: np.ndarray,
    variant_position: np.ndarray,
    meta_data: pandas.DataFrame,
    sample_id: List[str],
    sample_missing_threshold: float = 0.05,
    variant_missing_threshold: float = 0.05,
) -> Tuple[allel.GenotypeArray, pandas.DataFrame, List[str], np.ndarray, np.ndarray]:
    """
    Performs iterative filtering of a GenotypeArray based on missing data
    in both samples and variants, saving the intermediate results.

    This function repeatedly applies stricter missingness filters to the
    genotype array until a specified threshold for both samples and variants
    is reached.

    Args:
        gt (allel.GenotypeArray): The input genotype array to be filtered.
        sample_missing_threshold (float, optional): The target maximum
            proportion of missing data allowed for a variant. The filtering
            stops once the current threshold reaches this value. Defaults to 0.2.
        variant_missing_threshold (float, optional): The target maximum
            proportion of missing data allowed for a sample. The filtering
            stops once the current threshold reaches this value. Defaults to 0.2.
    """
    missingness_path = f"../data/missingness/pf8_nigerian_genotype_missingness_filter_{sample_missing_threshold}_{variant_missing_threshold}.npy"
    chromosome_path = f"../data/missingness/pf8_nigerian_chromosome_missingness_filter_{sample_missing_threshold}_{variant_missing_threshold}.npy"
    position_path = f"../data/missingness/pf8_nigerian_position_missingness_filter_{sample_missing_threshold}_{variant_missing_threshold}.npy"
    metadata_path = f"../data/missingness/pf8_metadata_filter_{sample_missing_threshold}_{variant_missing_threshold}.csv"
    sampleid_path = f"../data/missingness/pf8_sampleid_filter_{sample_missing_threshold}_{variant_missing_threshold}.npy"
    missing_sample_porportion = 0.9
    missing_variant_porportion = 0.9

    if (
        os.path.exists(missingness_path)
        and os.path.exists(chromosome_path)
        and os.path.exists(position_path)
        and os.path.exists(metadata_path)
    ):
        # NOTE: continuing for now, so that we can
        # also filter the sample metadata
        logger.info("Using cached filtered missingness")
        print(f"shape of original data {gt.shape}")
        snp_data = np.load(missingness_path)
        gt = allel.GenotypeArray(snp_data)
        print(f"shape of loaded data {gt.shape}")
        c = np.load(chromosome_path, allow_pickle=True)
        p = np.load(position_path, allow_pickle=True)
        m = pandas.read_csv(metadata_path)
        s = np.load(sampleid_path, allow_pickle=True)
        print(f"shape of loaded chromosome {c.shape}")
        print(f"shape of loaded position {p.shape}")
        return gt, m, s, c, p

    print(f"The shape of the metadata is {meta_data.shape}")
    print(f"The shape of the sample id is {len(sample_id)}")

    print(f"overall missingness {gt.count_missing()}")

    # iteratively filter out samples and variants with
    # up until missing porportion of 0.2
    while (
        missing_sample_porportion > sample_missing_threshold
        or missing_variant_porportion > variant_missing_threshold
    ):
        sample_count = gt.n_samples
        variant_count = gt.n_variants

        if missing_sample_porportion > sample_missing_threshold:
            sample_missing_per_variant = gt.count_missing(axis=1)
            filter = (
                sample_missing_per_variant / sample_count
            ) <= missing_sample_porportion
            if len(filter) > 0:
                gt = gt[filter]
                variant_chromosome = variant_chromosome[filter]
                variant_position = variant_position[filter]
            missing_sample_porportion = round(missing_sample_porportion - 0.1, 1)

        if missing_variant_porportion > variant_missing_threshold:
            variant_missing_per_sample = gt.count_missing(axis=0)
            filter = (
                variant_missing_per_sample / variant_count
            ) <= missing_variant_porportion
            if len(filter) > 0:
                gt = gt[:, filter, :]
                sample_id = sample_id[filter]
                meta_data = meta_data.loc[filter, :]

            missing_variant_porportion = round(missing_variant_porportion - 0.1, 1)

    print(f"overall missingness {gt.count_missing()}")
    np.save(missingness_path, gt)
    np.save(chromosome_path, variant_chromosome)
    np.save(position_path, variant_position)
    np.save(sampleid_path, sample_id)

    meta_data.to_csv(metadata_path, index=False)

    print(f"The filtered shape of the metadata is {meta_data.shape}")
    print(f"The filtered shape of the sample id is {len(sample_id)}")
    print(f"The filtered genotype shape is {gt.shape}")
    print(f"The filtered chromosome shape is {variant_chromosome.shape}")
    print(f"The filtered position shape is {variant_position.shape}")
    return gt, meta_data, sample_id, variant_chromosome, variant_position


def filter_mac(
    genotype_data: allel.GenotypeArray,
    variant_chromosome: np.ndarray,
    variant_position: np.ndarray,
    mac=5,
) -> Tuple[allel.GenotypeArray, np.ndarray, np.ndarray]:

    ac = genotype_data.count_alleles()
    mac_filter = ac[:, 1:] >= mac
    mac_filter = [any(filter) for filter in mac_filter]
    print(f"Total variant passing check is {sum(mac_filter)}")
    genotype_data = genotype_data[mac_filter]
    variant_chromosome = variant_chromosome[mac_filter]
    variant_position = variant_position[mac_filter]
    return genotype_data, variant_chromosome, variant_position


def filter_maf(
    genotype_data: allel.GenotypeArray,
    variant_chromosome: np.ndarray,
    variant_position: np.ndarray,
    maf=0.01,
) -> Tuple[allel.GenotypeArray, np.ndarray, np.ndarray]:
    ac = genotype_data.count_alleles()
    af = ac.to_frequencies()
    maf_filter = af[:, 1:] >= maf
    maf_filter = [any(filter) for filter in maf_filter]
    print(f"Total variant passing check is {sum(maf_filter)}")
    genotype_data = genotype_data[maf_filter]
    variant_chromosome = variant_chromosome[maf_filter]
    variant_position = variant_position[maf_filter]
    return genotype_data, variant_chromosome, variant_position


def main():
    logger.info("--- Starting temporal genomics pipeline ---")
    snp_dataset, pf8_metadata, sample_ids = retrieve_data.main()

    genotype_data, variant_chromosome, variant_position = retrieve_genotype(
        snp_dataset,
    )
    print("chromosome shape ", variant_chromosome.shape)
    print("genotype shape ", genotype_data.shape)
    print("position shape", variant_position.shape)
    genotype_data, pf8_metadata, sample_ids, variant_chromosome, variant_position = (
        filter_missing_genotype(
            genotype_data,
            variant_chromosome,
            variant_position,
            pf8_metadata,
            sample_ids,
        )
    )
    genotype_data, variant_chromosome, variant_position = filter_mac(
        genotype_data, variant_chromosome, variant_position
    )
    genotype_data, variant_chromosome, variant_position = filter_maf(
        genotype_data, variant_chromosome, variant_position
    )
    # dra.main(
    #    genotype_data, variant_chromosome, variant_position, pf8_metadata, sample_ids
    # )
    # sa.main(genotype_data, variant_chromosome, variant_position)
    # nea.main(genotype_data, variant_position, pf8_metadata, sample_ids)

    genotype_data, variant_chromosome, variant_position = ld.prune_ld(
        genotype_data, variant_position, variant_chromosome
    )
    genotype_data = pca.main(pf8_metadata, genotype_data, sample_ids)

    logger.info("----- Temporal genomics pipeline completed -----")


if __name__ == "__main__":
    main()
