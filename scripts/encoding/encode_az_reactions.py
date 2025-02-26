# %%
import pickle
import gzip
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from drfp import DrfpEncoder
from tqdm import tqdm


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
    df["smiles"] = df.reactant_smiles + "." + df.solvent_smiles + "." + df.base_smiles + ">>" + df.product_smiles

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

output = []

for fold_idx in tqdm(range(10)):
    root_path = Path(__file__).resolve().parent
    az_path = Path(root_path, "../../data/az")

    train, valid, test = get_az_rxns(fold_idx)

    output_splits = {}
    for data, split in [(train, "train"), (valid, "valid"), (test, "test")]:
        X, mapping = DrfpEncoder.encode(
            data.smiles.to_numpy(),
            n_folded_length=2048,
            radius=3,
            rings=True,
            mapping=True,
            include_hydrogens=True,
        )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y = data["yield"].to_numpy()

        # X_props = data.drop(
        #     columns=[
        #         "yield",
        #         "reactant_smiles",
        #         "solvent_smiles",
        #         "base_smiles",
        #         "product_smiles",
        #         "id",
        #         "smiles",
        #     ]
        # ).to_numpy()
        #
        # X = np.concatenate((X, X_props), axis=1)

        output_splits[split] = {
            "X": X,
            "y": y,
            "mapping": mapping,
            "smiles": data.smiles.to_numpy(),
        }

    output.append(output_splits)

out_file_name = Path(az_path, f"az-2048-3-true.pkl")
out_file_name_gz = Path(az_path, f"az-2048-3-true.pkl.gz")

with open(out_file_name, "wb+") as f:
    pickle.dump(output, f)

with gzip.open(out_file_name_gz, "wb+") as f:
    pickle.dump(output, f)
