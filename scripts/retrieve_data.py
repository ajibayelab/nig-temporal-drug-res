#!/usr/bin/env python3
"""
MalariaGen Sample Metadata Extraction Script

This script retrieves sample metadata from the MalariaGen database using the
malariagen_data package and exports it to a CSV file for further analysis.

Dependencies:
    - malariagen_data: Main package for accessing MalariaGen data
    - pandas: Data manipulation and CSV export
    - numpy: Numerical operations (if needed)

Usage:
    python retrieve_data.py [options]

Author: Isong Josiah
Date: 2025-01-09
Version: 1.0
"""

import logging as log
import sys
from typing import List

from pandas import DataFrame
from xarray import Dataset


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


def main() -> tuple[Dataset, DataFrame, List[str]]:
    import malariagen_data
    import pandas as pd

    # extract plasmodium falciparum 8 data
    logger.info("Connecting with malariagen to retrieve plasmodium falciparum 8 data")
    pf8_data = malariagen_data.Pf8()
    pf8_raw_metadata = pf8_data.sample_metadata()
    pf8_metadata = pf8_data.sample_metadata()

    # infer sample metadata to only include sample from Nigeria
    # that passes quality checks
    logger.info("Filtering plasmodium falciparum data by country and QC status")
    pf8_metadata = pf8_metadata[
        (pf8_metadata["QC pass"]) & (pf8_metadata["Country"] == "Nigeria")
    ]
    logger.info(f"unique countries in sample metadata after filtering -> {pd.unique(pf8_metadata["Country"])[0]}")


    # load inferred drug resistance status classification
    logger.info("Retrieving plasmodium falciparum drug resistance data")
    pf8_resistance_data = pd.read_csv(
        'https://pf8-release.cog.sanger.ac.uk/Pf8_inferred_resistance_status_classification.tsv', sep='\t'
    ).rename(columns={'sample': 'Sample'})

    # combine sample metadata to inferred drug resistance and save the data as a 
    # csv file
    logger.info("Merging sample metadata with drug resistance data")
    pf8_resistance_data = pd.merge(
        left=pf8_metadata,
        right=pf8_resistance_data,
        left_on='Sample',
        right_on='Sample',
        how='inner'
    )
    # TODO: uncomment when you figure out python modules
    # pf8_resistance_data.to_csv("../data/pf8_nigeria_drug_resistance.csv")

    # save zarr and attempt to convert to vcf
    logger.info("Generate Xarray Dataset")
    pf8_snp_only= malariagen_data.Pf8("s3://pf8-release/snp-only/")
    pf8_variant_ds = pf8_snp_only.variant_calls(extended=True)
    loc_filter = (pf8_raw_metadata["QC pass"] == True) & (pf8_raw_metadata["Country"] == "Nigeria").values
    ds = pf8_variant_ds.isel(samples=loc_filter)
    logger.info("Xarray Dataset generated")

    # TODO: cast object type to improve performance for writing
    # to disk
    return ds, pf8_resistance_data, pf8_resistance_data["Sample"].values


if __name__ == "__main__":
    logger.info("Retrieve data cannot be run as a script right now")

