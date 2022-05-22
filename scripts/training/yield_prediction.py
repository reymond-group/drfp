import pickle
from pathlib import Path
from typing import Tuple
from statistics import stdev
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


def save_results(
    set_name: str,
    split_id: str,
    file_name: str,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    with open(f"{set_name}_{split_id}_{file_name}.csv", "w+") as f:
        for gt, pred in zip(ground_truth, prediction):
            f.write(f"{set_name},{split_id},{file_name},{gt},{pred}\n")


def load_data(
    path: str, valid_frac: str = 0.1, split=2767
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = pickle.load(
        open(
            path,
            "rb",
        )
    )
    print(len(X))
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


def predict_buchwald_hartwig_cv():
    splits = [98, 197, 395, 791, 1186, 1977, 2766]
    buchwald_hartwig_yield_root = Path("../../data/")

    buchwald_hartwig_yield_fps = []

    for i in range(1, 11):
        buchwald_hartwig_yield_fps.append(f"FullCV_{str(i).zfill(2)}-2048-3-true.pkl")

    for split in splits:
        r2s = []
        for sample_file in buchwald_hartwig_yield_fps:
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
                Path(buchwald_hartwig_yield_root, sample_file), split=split
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

            y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
            y_pred[y_pred < 0.0] = 0.0

            save_results("buchwald_hartwig_cv", split, sample_file, y_test, y_pred)
            r_squared = r2_score(y_test, y_pred)
            r2s.append(r_squared)
        print(f"{100 * round(split / 3955, 3)}%", sum(r2s) / len(r2s), stdev(r2s))


def predict_buchwald_hartwig_tests():
    buchwald_hartwig_yield_root = Path("../../data/")

    buchwald_hartwig_yield_fps = []

    for i in range(1, 5):
        buchwald_hartwig_yield_fps.append(f"Test{i}-2048-3-true.pkl")

    r2s_all = []
    for i, sample_file in enumerate(buchwald_hartwig_yield_fps):
        r2s = []
        for seed in [42, 69, 2222, 2626]:
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
                Path(buchwald_hartwig_yield_root, sample_file), split=3058
            )

            model = XGBRegressor(
                n_estimators=999999,
                learning_rate=0.01,
                max_depth=12,
                min_child_weight=6,
                colsample_bytree=0.6,
                subsample=0.8,
                random_state=seed,
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

            save_results("buchwald_hartwig_tests", sample_file, seed, y_test, y_pred)
            r_squared = r2_score(y_test, y_pred)
            r2s_all
            r2s.append(r_squared)

        print(f"Test {i + 1}", sum(r2s) / len(r2s), stdev(r2s))


def predict_suzuki_miyaura():
    splits = [4032]
    buchwald_hartwig_yield_root = Path("../../data/")

    files = []

    for i in range(10):
        files.append(f"random_split_{i}-2048-3-true.pkl")

    for split in splits:
        r2s = []
        for i, sample_file in enumerate(files):
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
                Path(buchwald_hartwig_yield_root, sample_file), split=split
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

            y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
            y_pred[y_pred < 0.0] = 0.0

            save_results("suzuki_miyaura", split, sample_file, y_test, y_pred)
            r_squared = r2_score(y_test, y_pred)
            print(f"Test {i + 1}", r_squared)
            r2s.append(r_squared)

        print("Test average:", sum(r2s) / len(r2s), stdev(r2s))


def main():
    predict_buchwald_hartwig_cv()
    # predict_buchwald_hartwig_tests()
    # predict_suzuki_miyaura()


if __name__ == "__main__":
    main()
