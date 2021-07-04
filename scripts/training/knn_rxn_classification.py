# Adapted from: https://github.com/rxn4chemistry/rxnfp/blob/master/nbs/10_results_uspto_1k_tpl.ipynb

import pickle
import click
import faiss
import numpy as np
from pathlib import Path
from typing import Optional
from collections import Counter
from pycm import ConfusionMatrix


def get_nearest_neighbours_prediction(
    train_X: np.array, train_y: np.array, eval_X: np.array, n_neighbours: int = 5
) -> list:
    """
    Use faiss to make a K-nearest neighbour prediction
    """
    # Indexing
    index = faiss.IndexFlatL2(len(train_X[0]))
    index.add(train_X.astype(np.float32))

    # Querying
    _, results = index.search(eval_X.astype(np.float32), n_neighbours)

    # Scoring
    y_pred = get_pred(train_y, results)

    return y_pred


def get_pred(y: list, results: list) -> list:
    """
    Get most common label from nearest neighbour list
    """
    y_pred = []
    for i, r in enumerate(results):
        y_pred.append(Counter(y[r]).most_common(1)[0][0])
    return y_pred


def get_cache_confusion_matrix(
    name: str, actual_vector: list, predict_vector: list
) -> ConfusionMatrix:
    """
    Make confusion matrix and save it.
    """
    cm_cached = load_confusion_matrix(f"{name}.pickle")

    if cm_cached is not None:
        return cm_cached

    cm = ConfusionMatrix(actual_vector=actual_vector, predict_vector=predict_vector)
    cm.save_html(name)
    with open(f"{name}.pickle", "wb") as f:
        pickle.dump(cm, f)
    return cm


def load_confusion_matrix(path: str) -> Optional[ConfusionMatrix]:
    """
    Load confusion matrix if existing.
    """
    if Path(path).is_file():
        return pickle.load(open(path, "rb"))
    return None


@click.command()
@click.argument("input_train_filepath", type=click.Path(exists=True))
@click.argument("input_test_filepath", type=click.Path(exists=True))
@click.option("--cm-name", type=str, default="cm")
@click.option("--reduce", type=float, default=1.0)
def main(input_train_filepath, input_test_filepath, cm_name: str, reduce: float):
    X_train, y_train, _ = pickle.load(open(input_train_filepath, "rb"))
    X_test, y_test, _ = pickle.load(open(input_test_filepath, "rb"))

    # Reduce training set size
    train_indices = np.random.choice(
        np.arange(len(X_train)), int(reduce * len(X_train)), replace=False
    )

    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    y_pred = [
        int(i) for i in get_nearest_neighbours_prediction(X_train, y_train, X_test)
    ]

    cm = get_cache_confusion_matrix(
        cm_name,
        y_test,
        y_pred,
    )

    print(f"Accuracy : {cm.overall_stat['Overall ACC']:.3f}")
    print(f"MCC : {cm.overall_stat['Overall MCC']:.3f}")
    print(f"CEN : {cm.overall_stat['Overall CEN']:.3f}")


if __name__ == "__main__":
    main()
