import pickle
from pathlib import Path
from typing import Tuple
from statistics import stdev
import numpy as np
from xgboost import XGBRegressor


def load_data(
    path: str, valid_frac: str = 0.1, split=3955
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = pickle.load(
        open(
            path,
            "rb",
        )
    )

    X_test = X[split:]
    X_train = X[:split]

    y_test = y[split:]
    y_train = y[:split]

    np.random.seed(42)
    valid_indices = np.random.choice(
        np.arange(len(X_train)), int(valid_frac * len(X_train)), replace=False
    )

    X_valid = X_train[valid_indices]
    y_valid = y_train[valid_indices]

    train_indices = list(set(range(len(X_train))) - set(valid_indices))

    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def buchwald_hartwig_model():
    buchwald_hartwig_yield_root = Path("../../data/")
    buchwald_hartwig_yield_models = Path("../../models/")

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        Path(buchwald_hartwig_yield_root, "FullCV_01-2048-3-true.pkl")
    )

    model = XGBRegressor(
        n_estimators=999999,
        learning_rate=0.01,
        max_depth=12,
        min_child_weight=6,
        colsample_bytree=0.6,
        subsample=0.8,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=20,
        verbose=False,
    )

    model.save_model(
        Path(buchwald_hartwig_yield_models, "buchwald-hartwig-xgb-model.json")
    )


def main():
    buchwald_hartwig_model()


if __name__ == "__main__":
    main()
