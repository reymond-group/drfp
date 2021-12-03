# Adapted from: https://github.com/rxn4chemistry/rxnfp/blob/master/nbs/10_results_uspto_1k_tpl.ipynb

import pickle
import click
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neural_network import MLPClassifier
import numpy as np
from pathlib import Path
from typing import Optional
from collections import Counter
from pycm import ConfusionMatrix
from sklearn.preprocessing import LabelEncoder


def get_pred(
    train_X: np.array, train_y: np.array, eval_X: np.array, n_classes: int
) -> list:
    """
    Get predictaions using a simple MLP.
    """

    model = keras.models.Sequential(
        [
            keras.layers.Dense(1664, input_dim=len(train_X[0]), activation=tf.nn.tanh),
            keras.layers.Dense(n_classes, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    model.fit(
        train_X,
        train_y,
        epochs=10,
        batch_size=64,
    )

    return np.argmax(model.predict(eval_X), axis=-1)


def get_cache_confusion_matrix(
    name: str, actual_vector: list, predict_vector: list
) -> ConfusionMatrix:
    """
    Make confusion matrix and save it.
    """
    cm = ConfusionMatrix(actual_vector=actual_vector, predict_vector=predict_vector)
    cm.save_html(name)
    with open(f"{name}.pickle", "wb") as f:
        pickle.dump(cm, f)
    return cm


@click.command()
@click.argument("input_train_filepath", type=click.Path(exists=True))
@click.argument("input_test_filepath", type=click.Path(exists=True))
def main(input_train_filepath, input_test_filepath):
    X_train, y_train, _ = pickle.load(open(input_train_filepath, "rb"))
    X_test, y_test, _ = pickle.load(open(input_test_filepath, "rb"))

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    n_classes = len(le.classes_)

    # Reduce training set size
    train_indices = np.random.choice(
        np.arange(len(X_train)), int(1.0 * len(X_train)), replace=False
    )

    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    print(y_train)
    y_pred = get_pred(X_train, y_train, X_test, n_classes)

    cm = get_cache_confusion_matrix(
        "drfp-tpl",
        y_test,
        y_pred,
    )

    print(f"Accuracy : {cm.overall_stat['Overall ACC']:.3f}")
    print(f"MCC : {cm.overall_stat['Overall MCC']:.3f}")
    print(f"CEN : {cm.overall_stat['Overall CEN']:.3f}")


if __name__ == "__main__":
    main()
