import logging as log
import sys

import allel
from pandas.core import missing
from pandas.core.generic import sample
from xarray import Dataset
import numpy as np

import retrieve_data
import population_structure.neighborhood_joining_tree as njt

import population_structure.pcoa as pcoa
import population_structure.linkage_disequilibrum as ld

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


def retrieve_genotype(snp_dataset: Dataset) -> allel.GenotypeArray:
    import os

    logger.info("--- Retrieving genotype information ---")
    genotype_path = "../data/pf8_nigerian_genotype_filtered.npyoo"
    o_genotype_path = "../data/pf8_nigerian_genotype.npy"
    chrom_path = "../data/pf8_nigeria_variant_chrom.npy"
    snp_data = None
    gt: allel.GenotypeArray
    if os.path.exists(genotype_path):
        logger.info("--- Getting genotype from file ---")
        snp_data = np.load(genotype_path)
        gt = allel.GenotypeArray(snp_data)

        ac = gt.count_alleles()

        # filter for only biallelic alleles
        biallelic_filter = ac.is_biallelic()
        chrom = np.load(chrom_path, allow_pickle=True)
        print(f"shape before filtering {chrom.shape}")
        chrom = chrom[biallelic_filter]
        print(f"shape after filtering {chrom.shape}")
        print(chrom)
    else:
        try:
            logger.info("--- Getting genotype from s3 ---")
            snp_data = snp_dataset.call_genotype
            gt = allel.GenotypeArray(snp_data)
            ac = gt.count_alleles()

            # filter for only biallelic alleles
            biallelic_filter = ac.is_biallelic()
            gt = gt[biallelic_filter]
            gt = filter_missing_genotype(gt)

            if not os.path.exists(chrom_path):
                chrom = snp_dataset["variant_chrom"]
                np.save(chrom_path, chrom)
            chrom = np.load(chrom_path, allow_pickle=True)
            print(f"shape before filtering {chrom.shape}")
            chrom = chrom[biallelic_filter]
            print(f"shape after filtering {chrom.shape}")
            print(chrom)

            logger.info("--- Saving genotype to file ---")
            np.save(genotype_path, gt)
        except Exception as e:
            logger.fatal(f"failed to retrieve genotype from s3")
            exit(1)

    if snp_data is None:
        logger.fatal(f"genotype content is none")
        exit(1)

    logger.info("--- Retrieved genotype information ---")
    return gt


def filter_missing_genotype(
    gt: allel.GenotypeArray,
    sample_missing_threshold: float = 0.05,
    variant_missing_threshold: float = 0.05,
) -> allel.GenotypeArray:
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
    ofp = "../data/pf8_nigerian_genotype_filtered"
    missing_sample_porportion = 0.9
    missing_variant_porportion = 0.9

    print(gt.shape)
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
                print(gt.shape)
                fp = f"{ofp}_sample_filter_{missing_sample_porportion}.npy"
                np.save(fp, gt)
            missing_sample_porportion = round(missing_sample_porportion - 0.1, 1)

        if missing_variant_porportion > variant_missing_threshold:
            variant_missing_per_sample = gt.count_missing(axis=0)
            filter = (
                variant_missing_per_sample / variant_count
            ) <= missing_variant_porportion
            if len(filter) > 0:
                gt = gt[:, filter, :]
                print(gt.shape)
                fp = f"{ofp}_variant_filter_{missing_sample_porportion}.npy"
                np.save(fp, gt)
            missing_variant_porportion = round(missing_variant_porportion - 0.1, 1)

    print(f"overall missingness {gt.count_missing()}")
    print(gt.shape)
    return gt


def filter_mac(gt: allel.GenotypeArray, mac_threshold=1) -> allel.GenotypeArray:
    logger.info("--- Filtering by Minor Allele Count ---")
    ac = gt.count_alleles()
    # filter out zero counts
    ac = np.array([row[row != 0] for row in ac])
    mac = ac.min(axis=1)
    mac_filter = mac > mac_threshold
    print(np.sum(mac_filter))
    return gt[mac_filter]


def filter_maf(gt: allel.GenotypeArray, maf_threshold=0.01) -> allel.GenotypeArray:
    logger.info("--- Filtering by Minor Allele Frequency ---")
    ac = gt.count_alleles()
    af = ac.to_frequencies()
    # filter out zero frequencies
    af = np.array([row[row != 0] for row in af])
    maf = af.min(axis=1)
    maf_filter = maf >= maf_threshold
    print(np.sum(maf_filter))
    return gt[maf_filter]


def main():
    logger.info("--- Starting temporal genomics pipeline ---")

    snp_dataset, pf8_metadata, sample_ids = retrieve_data.main()
    chrom = snp_dataset["variant_chrom"]
    print(chrom.shape)
    np.save("../data/pf8_nigeria_variant_genotype.npy", chrom)
    genotype_data = retrieve_genotype(snp_dataset)
    genotype_data = filter_maf(genotype_data)
    genotype_data = filter_mac(genotype_data, mac_threshold=5)
    genotype_data = pcoa.main(pf8_metadata, genotype_data, sample_ids)
    ld.prune_ld(genotype_data)

    logger.info("--- Temporal genomics pipeline completed ---")


if __name__ == "__main__":
    main()
