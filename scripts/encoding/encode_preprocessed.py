# -*- coding: utf-8 -*-
import os
import pickle
import click
import logging
import multiprocessing
from functools import partial
from typing import Iterable

from drfp import DrfpEncoder

import numpy as np
import pandas as pd


def encode(smiles: Iterable, length: int = 2048, radius: int = 3) -> np.ndarray:
    return DrfpEncoder.encode(
        smiles,
        n_folded_length=length,
        radius=radius,
        rings=True,
    )


def encode_dataset(smiles: Iterable, length: int, radius: int) -> np.ndarray:
    """Encode the reaction SMILES to drfp"""

    cpu_count = (
        multiprocessing.cpu_count()
    )  # Data gets too big for piping when splitting less in python < 2.8

    # Split reaction SMILES for multiprocessing
    k, m = divmod(len(smiles), cpu_count)
    smiles_chunks = (
        smiles[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(cpu_count)
    )

    # Run the fingerprint generation in parallel
    results = []
    with multiprocessing.Pool(cpu_count) as p:
        results = p.map(partial(encode, length=length, radius=radius), smiles_chunks)

    return np.array([item for s in results for item in s])


def add_split_to_filepath(filepath: str, split: str) -> str:
    name, ext = os.path.splitext(filepath)
    return f"{name}_{split}{ext}"


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--cols", nargs=3, type=str, required=True)
@click.option("--sep", type=str, default=",")
@click.option("--length", type=int, default=2048)
@click.option("--radius", type=int, default=3)
def main(input_filepath, output_filepath, cols, sep, length, radius):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    df = pd.read_csv(input_filepath, sep=sep, usecols=list(cols))

    splits = df[cols[2]].unique()

    for s in splits:
        split_filepath = add_split_to_filepath(output_filepath, s)

        if os.path.exists(split_filepath):
            print(f"{split_filepath} already exists.")
            continue

        print(f"Encoding {split_filepath}...")

        df_split = df[df[cols[2]] == s]

        print(f"{len(df_split)} reactions...")

        smiles = df_split[cols[0]].to_numpy()
        y = df_split[cols[1]].to_numpy()

        logger.info(f"generating drfp fingerprints ({s})")
        X = encode_dataset(smiles, length, radius)

        logger.info(f"pickling {s} data set to {split_filepath}")
        with open(split_filepath, "wb+") as f:
            pickle.dump((X, y, smiles), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
