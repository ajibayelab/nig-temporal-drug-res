import logging as log
from allel import AlleleCountsArray, GenotypeArray
from typing import List
from sys import stdout

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        log.FileHandler("pcoa.log"),
        log.StreamHandler(stdout),
    ],
)
logger = log.getLogger(__name__)


def main(ga: GenotypeArray, pop: List) -> float:
    import allel
    from numpy import sum, nan_to_num

    a, b, c = allel.weir_cockerham_fst(ga, pop)
    a, b, c = nan_to_num(a), nan_to_num(b), nan_to_num(c)
    fst = sum(a) / (sum(a) + sum(b) + sum(c))
    print("weir_cockerham_fst is ->", fst)

    ac1 = ga.count_alleles(subpop=pop[0])
    ac2 = ga.count_alleles(subpop=pop[1])
    num, den = allel.hudson_fst(ac1, ac2)
    num, den = nan_to_num(num), nan_to_num(den)
    avg_fst = sum(num) / sum(den)
    return avg_fst
