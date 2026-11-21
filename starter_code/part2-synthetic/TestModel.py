import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def load_xy(df):
    # replace nan with -99
    df = df.fillna(-99)
    target = "Loan Status"
    synthetic = df.pop("Synthetic").astype(bool).to_numpy() if "Synthetic" in df else np.zeros(len(df), dtype=bool)
    x_df = df.drop(target, axis=1)
    y_df = df[target]

    y_df = LabelEncoder().fit_transform(y_df)
    return x_df, y_df, synthetic

def run_test(x_df, y_df, synthetic, synthetic_fraction=1.0, full_dataset_split=False):
    # The primary mode holds out only real records. The optional full-dataset
    # mode stratifies across every real and synthetic row for a supplemental check.
    eligible_idx = np.arange(len(y_df)) if full_dataset_split else np.flatnonzero(~synthetic)
    train_idx, remainder_idx = train_test_split(
        eligible_idx, test_size=0.35, random_state=42, stratify=y_df[eligible_idx]
    )
    calibration_idx, test_idx = train_test_split(
        remainder_idx, test_size=(4 / 7), random_state=43,
        stratify=y_df[remainder_idx],
    )
    synthetic_idx = np.flatnonzero(synthetic) if not full_dataset_split else train_idx[synthetic[train_idx]]
    if synthetic_fraction < 1.0 and len(synthetic_idx):
        rng = np.random.default_rng(42)
        keep = max(1, int(len(synthetic_idx) * synthetic_fraction))
        synthetic_idx = np.sort(rng.choice(synthetic_idx, size=keep, replace=False))
    if full_dataset_split:
        real_train_idx = train_idx[~synthetic[train_idx]]
        fit_idx = np.concatenate([real_train_idx, synthetic_idx])
        x_train, y_train = x_df.iloc[fit_idx], y_df[fit_idx]
    else:
        x_train = pd.concat([x_df.iloc[train_idx], x_df.iloc[synthetic_idx]], ignore_index=True)
        y_train = np.concatenate([y_df[train_idx], y_df[synthetic_idx]])
    x_calibration, y_calibration = x_df.iloc[calibration_idx], y_df[calibration_idx]
    x_test, y_test = x_df.iloc[test_idx], y_df[test_idx]
    # Scaling plus early stopping keeps training practical for both original and
    # augmented data while ensuring metrics come from held-out examples.
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=0.001, batch_size=256, learning_rate_init=0.001,
            max_iter=100, early_stopping=True, n_iter_no_change=8,
            random_state=42, verbose=False,
        ),
    )
    clf.fit(x_train, y_train)
    calibration_prob = clf.predict_proba(x_calibration)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_calibration, calibration_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    threshold = float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5
    y_pred = (clf.predict_proba(x_test)[:, 1] >= threshold).astype(int)
    report = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0)
    test_kind = "full-dataset split" if full_dataset_split else "real-only held-out"
    print(f"{test_kind} accuracy: {accuracy_score(y_test, y_pred):.4f}; calibrated threshold: {threshold:.4f}")
    if full_dataset_split:
        print(f"Test split composition: {int((~synthetic[test_idx]).sum()):,} real + "
              f"{int(synthetic[test_idx].sum()):,} synthetic rows")
    print("Test results:")
    print(report)
    return report

def test_model(datapath, synthetic_fraction=1.0, full_dataset_split=False):
    df_base = pd.read_csv(datapath, sep=',')
    x_df, y_df, synthetic = load_xy(df_base)
    return run_test(
        x_df, y_df, synthetic, synthetic_fraction=synthetic_fraction,
        full_dataset_split=full_dataset_split,
    )
