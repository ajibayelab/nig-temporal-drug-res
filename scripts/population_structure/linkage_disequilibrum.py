from common import setup_logger
from numpy import ndarray
from allel import GenotypeArray
from typing import Tuple


def prune_ld(
    genotype_data: GenotypeArray, position: ndarray, chromosome: ndarray
) -> Tuple[GenotypeArray, ndarray, ndarray]:
    import allel

    gn = genotype_data.to_n_alt()
    prune_mask = allel.locate_unlinked(gn, size=50)

    print(f"prior genotype data {genotype_data.shape}")
    genotype_data = genotype_data.compress(prune_mask, axis=0)
    print(f"genotype data after mask {genotype_data.shape}")

    print(f"prior position {position.shape}")
    position = position.compress(prune_mask)
    print(f"position {position.shape}")

    print(f"prior chromosome {chromosome.shape}")
    chromosome = chromosome.compress(prune_mask)
    print(f"chromosome {chromosome.shape}")
    return genotype_data, chromosome, position


def main(genotype_data: GenotypeArray):
    import allel
    import numpy as np

    logger = setup_logger(__name__)
    logger.info("Starting Linkage Disequilibrum Analysis")

    # convert ndarray to allele.GenotypeArray
    ac = genotype_data.count_alleles()
    logger.info(f"allele count {ac}")

    gn = genotype_data.to_n_alt()
    r = allel.rogers_huff_r(gn)
    logger.info(f"linkage disequilibrum -> {r}")
    logger.info("Completed Linkage Disequilibrum Analysis")


if __name__ == "__main__":
    pass
