import pickle
from pathlib import Path
from typing import Tuple
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def save_results(set_name: str, split_id: str, file_name: str, ground_truth: np.ndarray, prediction: np.ndarray) -> None:
    with open(f"{set_name}_{split_id}_{file_name}.csv", "w+") as f:
        for gt, pred in zip(ground_truth, prediction):
            f.write(f"{set_name},{split_id},{file_name},{gt},{pred}\n")

def load_data(
    path_train: str,
    path_test: str,
    valid_frac: str = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, _ = pickle.load(
        open(
            path_train,
            "rb",
        )
    )

    subset_indices = np.random.choice(
        np.arange(len(X_train)), int(1.0 * len(X_train)), replace=False
    )
    X_train = X_train[subset_indices]
    y_train = y_train[subset_indices]

    X_test, y_test, _ = pickle.load(
        open(
            path_test,
            "rb",
        )
    )

    valid_indices = np.random.choice(
        np.arange(len(X_train)), int(valid_frac * len(X_train)), replace=False
    )
    X_valid = X_train[valid_indices]
    y_valid = y_train[valid_indices]

    train_indices = list(set(range(len(X_train))) - set(valid_indices))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def predict_uspto_above():
    uspto_root = Path("../../data/")

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        Path(uspto_root, "uspto_yields_above_2048_3_true_train.pkl"),
        Path(uspto_root, "uspto_yields_above_2048_3_true_test.pkl"),
    )

    model = XGBRegressor(
        n_estimators=999999,
        learning_rate=0.1,
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

    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
    y_pred[y_pred < 0.0] = 0.0

    save_results("uspto", "above", "above", y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    print(r_squared)


def predict_uspto_below():
    uspto_root = Path("../../data/")

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        Path(uspto_root, "uspto_yields_below_2048_3_true_train.pkl"),
        Path(uspto_root, "uspto_yields_below_2048_3_true_test.pkl"),
    )

    model = XGBRegressor(
        n_estimators=999999,
        learning_rate=0.1,
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

    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
    y_pred[y_pred < 0.0] = 0.0

    save_results("uspto", "below", "below", y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    print(r_squared)


def main():
    predict_uspto_above()
    predict_uspto_below()


if __name__ == "__main__":
    main()
