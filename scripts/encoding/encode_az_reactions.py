# %%
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from drfp import DrfpEncoder


# %%
def get_az_rxns(fold_idx: int = 0):
    """
    Convert the reactions described in the supplied files to reactino smiles
    """
    root_path = Path(__file__).resolve().parent
    az_path = Path(root_path, "../../data/az")
    splits = pickle.load(open(Path(az_path, "train_test_idxs.pickle"), "rb"))

    train_ids = splits["train_idx"][fold_idx + 1]
    test_ids = splits["test_idx"][fold_idx + 1]

    df = pd.read_csv(Path(az_path, "az_no_rdkit.csv"))
    df["smiles"] = (
        df.reactant_smiles
        + "."
        + df.solvent_smiles
        + "."
        + df.base_smiles
        + ">>"
        + df.product_smiles
    )

    train = df.iloc[train_ids]
    test = df.iloc[test_ids]

    # Validate on random sample from train
    valid = train.sample(frac=0.1)

    return train, valid, test


# %%
y_predictions = []
y_tests = []
r2_scores = []
rmse_scores = []

for fold_idx in range(10):
    root_path = Path(__file__).resolve().parent
    az_path = Path(root_path, "../../data/az")

    train, valid, test = get_az_rxns(fold_idx)

    for data, split in [(train, "train"), (valid, "valid"), (test, "test")]:
        X, mapping = DrfpEncoder.encode(
            data.smiles.to_numpy(),
            n_folded_length=2048,
            radius=3,
            rings=True,
            mapping=True,
        )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y = data["yield"].to_numpy()

        fingerprints_file_name = Path(az_path, f"{fold_idx}-{split}-2048-3-true.pkl")
        map_file_name = Path(az_path, f"{fold_idx}-{split}-2048-3-true.map.pkl")

        with open(map_file_name, "wb+") as f:
            pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(fingerprints_file_name, "wb+") as f:
            pickle.dump((X, y), f, protocol=pickle.HIGHEST_PROTOCOL)
