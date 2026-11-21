"""Compare classifiers on real held-out loans, optionally training with VAE rows."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent / "data"
REAL_PATH = DATA_DIR / "loan_continuous.csv"
SYNTHETIC_PATH = DATA_DIR / "loan_continuous_expanded.csv"


def split_real(real):
    y = real["Loan Status"].to_numpy(dtype=int)
    train_idx, remainder_idx = train_test_split(
        np.arange(len(real)), test_size=0.35, random_state=42, stratify=y
    )
    calibration_idx, test_idx = train_test_split(
        remainder_idx, test_size=4 / 7, random_state=43,
        stratify=y[remainder_idx],
    )
    return train_idx, calibration_idx, test_idx


def make_estimators():
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=500, random_state=42),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        ),
    }


def evaluate(name, estimator, x_train, y_train, x_cal, y_cal, x_test, y_test, weights=None):
    if weights is None:
        estimator.fit(x_train, y_train)
    else:
        estimator.fit(x_train, y_train, **{"sample_weight": weights})
    p_cal = estimator.predict_proba(x_cal)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_cal, p_cal)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    threshold = float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5
    p_test = estimator.predict_proba(x_test)[:, 1]
    y_pred = (p_test >= threshold).astype(int)
    print(f"\n{name}: threshold={threshold:.4f}; accuracy={accuracy_score(y_test, y_pred):.4f}; "
          f"ROC-AUC={roc_auc_score(y_test, p_test):.4f}; PR-AUC={average_precision_score(y_test, p_test):.4f}; "
          f"class-1 F1={f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))


def run(label, real, synthetic, train_idx, calibration_idx, test_idx, synthetic_fraction):
    feature_cols = [column for column in real.columns if column != "Loan Status"]
    x_real = real[feature_cols].fillna(-99).reset_index(drop=True)
    y_real = real["Loan Status"].to_numpy(dtype=int)
    x_train = x_real.iloc[train_idx].copy()
    y_train = y_real[train_idx].copy()

    if synthetic_fraction and len(synthetic):
        count = int(len(synthetic) * synthetic_fraction)
        selected = synthetic.sample(n=count, random_state=42)
        x_train = pd.concat([x_train, selected[feature_cols].fillna(-99)], ignore_index=True)
        y_train = np.concatenate([y_train, selected["Loan Status"].to_numpy(dtype=int)])

    x_cal = x_real.iloc[calibration_idx]
    y_cal = y_real[calibration_idx]
    x_test = x_real.iloc[test_idx]
    y_test = y_real[test_idx]

    # HistGradientBoosting has no class_weight argument; pass equivalent balanced
    # per-row weights. Other estimators use their built-in class_weight option.
    class_counts = np.bincount(y_train, minlength=2)
    hgb_weights = np.array([len(y_train) / (2 * class_counts[label]) for label in y_train])
    print(f"\n=== {label}: training rows={len(y_train):,} ===")
    for name, estimator in make_estimators().items():
        weights = hgb_weights if name == "Histogram gradient boosting" else None
        evaluate(name, estimator, x_train, y_train, x_cal, y_cal, x_test, y_test, weights)


def main():
    real = pd.read_csv(REAL_PATH).fillna(-99)
    expanded = pd.read_csv(SYNTHETIC_PATH).fillna(-99)
    if "Synthetic" not in expanded:
        raise ValueError("Expanded dataset must mark generated rows in the Synthetic column")
    synthetic = expanded.loc[expanded["Synthetic"].astype(bool)].drop(columns="Synthetic")
    train_idx, calibration_idx, test_idx = split_real(real)
    print(f"Real-only split: train={len(train_idx):,}, calibration={len(calibration_idx):,}, "
          f"test={len(test_idx):,}; test contains only original loans.")
    run("Original data", real, synthetic.iloc[0:0], train_idx, calibration_idx, test_idx, 0)
    run("Original + 25% of synthetic data", real, synthetic, train_idx, calibration_idx, test_idx, 0.25)
    run("Original + all synthetic data", real, synthetic, train_idx, calibration_idx, test_idx, 1.0)


if __name__ == "__main__":
    main()
