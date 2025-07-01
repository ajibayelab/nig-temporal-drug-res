from common import setup_logger
from numpy import ndarray
from allel import GenotypeArray


def prune_ld(genotype_data: GenotypeArray):
    import allel

    gn = genotype_data.to_n_alt()
    m = allel.locate_unlinked(gn)
    print(m)
    pass


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
