"""Inspect label quality and feature signal in the original loan dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent / "data" / "loan_continuous.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    target = "Loan Status"
    y = df[target].to_numpy()
    features = df.drop(columns=target)
    x = features.fillna(-99)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.25, random_state=42, stratify=y
    )

    print(f"Rows: {len(df):,}; columns including target: {df.shape[1]}")
    print("Target values:", df[target].value_counts(dropna=False).to_dict())
    print(f"Target prevalence for class 1: {(y == 1).mean():.3%}")
    print("Unexpected target values:", sorted(set(pd.unique(df[target])) - {0, 1}))

    feature_dupes = features.duplicated(keep=False)
    if feature_dupes.any():
        conflicts = features.loc[feature_dupes].assign(_target=y[feature_dupes])
        conflict_rows = int(conflicts.groupby(list(features.columns), dropna=False)["_target"].transform("nunique").gt(1).sum())
        print(f"Exact duplicate feature rows: {int(feature_dupes.sum()):,}; rows with conflicting labels: {conflict_rows:,}")
    else:
        print("Exact duplicate feature rows: 0; rows with conflicting labels: 0")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("Columns with missing values:", missing.to_dict())

    y_train, y_test = y[train_idx], y[test_idx]
    rows = []
    for column in features.columns:
        train_values = x.iloc[train_idx][column].to_numpy()
        test_values = x.iloc[test_idx][column].to_numpy()
        if len(np.unique(train_values)) < 2:
            continue
        train_auc = roc_auc_score(y_train, train_values)
        test_auc = roc_auc_score(y_test, test_values)
        rows.append((column, train_auc, test_auc, abs(test_auc - 0.5)))
    rows.sort(key=lambda row: row[3], reverse=True)
    print("\nTop single-feature test AUCs (direction shown by AUC < 0.5):")
    for column, train_auc, test_auc, _ in rows[:10]:
        print(f"{column}: train AUC={train_auc:.3f}, test AUC={test_auc:.3f}")

    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    models = {
        "Random forest": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        ),
    }
    weights = np.where(y_train == 1, len(y_train) / (2 * np.sum(y_train == 1)),
                       len(y_train) / (2 * np.sum(y_train == 0)))
    print("\nTrain-versus-test ROC-AUC:")
    for name, model in models.items():
        if name == "Histogram gradient boosting":
            model.fit(x_train, y_train, sample_weight=weights)
        else:
            model.fit(x_train, y_train)
        train_score = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
        test_score = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        print(f"{name}: train={train_score:.3f}, test={test_score:.3f}")


if __name__ == "__main__":
    main()
