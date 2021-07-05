# %%
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from drfp import DrfpEncoder

# %%
NAME_SPLIT = [
    ("random_split_0", 4032),
    ("random_split_1", 4032),
    ("random_split_2", 4032),
    ("random_split_3", 4032),
    ("random_split_4", 4032),
    ("random_split_5", 4032),
    ("random_split_6", 4032),
    ("random_split_7", 4032),
    ("random_split_8", 4032),
    ("random_split_9", 4032),
]

# %%
y_predictions = []
y_tests = []
r2_scores = []
rmse_scores = []

for (name, split) in NAME_SPLIT:
    fingerprints_file_name = f"../../data/{name}-2048-3-true.pkl"
    map_file_name = f"../../data/{name}-2048-3-true.map.pkl"

    df = pd.read_csv(f"../../data/Suzuki-Miyaura/random_splits/{name}.tsv", sep="\t")

    df = df[["rxn", "y"]]
    df.columns = ["text", "labels"]

    X, mapping = DrfpEncoder.encode(
        df.text.to_numpy(), n_folded_length=2048, radius=3, rings=True, mapping=True
    )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = df.labels.to_numpy()

    with open(map_file_name, "wb+") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(fingerprints_file_name, "wb+") as f:
        pickle.dump((X, y), f, protocol=pickle.HIGHEST_PROTOCOL)
