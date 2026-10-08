from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sepsispulse.features import FEATURE_NAMES, engineer_forecast_windows
from sepsispulse.schema import (
    FEATURE_VERSION,
    FORECAST_HORIZON_HOURS,
    PREDICTION_TASK,
    WINDOW_HOURS,
)

RANDOM_SEED = 42
NEGATIVE_SAMPLE_RATE = 0.2


def _patient_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep="|", na_values=["NaN"])
    features, labels = engineer_forecast_windows(
        frame, horizon_hours=FORECAST_HORIZON_HOURS
    )
    if len(features) != len(labels):
        raise ValueError(f"Feature/label length mismatch in {path}")
    valid = labels >= 0
    return (
        features.to_numpy(dtype=np.float32)[valid],
        labels[valid],
    )


def _iter_patient_batches(
    paths: list[Path],
    *,
    negative_sample_rate: float | None,
    batch_size: int = 128,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED)
    batch_features: list[np.ndarray] = []
    batch_labels: list[np.ndarray] = []
    patients_in_batch = 0

    for path in paths:
        features, labels = _patient_windows(path)
        if negative_sample_rate is not None and len(labels):
            keep = (labels == 1) | (rng.random(len(labels)) < negative_sample_rate)
            features = features[keep]
            labels = labels[keep]
        if len(labels):
            batch_features.append(features)
            batch_labels.append(labels)
        patients_in_batch += 1

        if patients_in_batch >= batch_size:
            if batch_features:
                yield np.concatenate(batch_features), np.concatenate(batch_labels)
            batch_features.clear()
            batch_labels.clear()
            patients_in_batch = 0

    if batch_features:
        yield np.concatenate(batch_features), np.concatenate(batch_labels)


def _collect_training_data(
    paths: list[Path], negative_sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    batches = list(
        _iter_patient_batches(
            paths, negative_sample_rate=negative_sample_rate
        )
    )
    if not batches:
        raise ValueError("No complete six-hour training windows were found.")
    features = np.concatenate([batch[0] for batch in batches])
    labels = np.concatenate([batch[1] for batch in batches])
    if len(np.unique(labels)) != 2:
        raise ValueError("Training data must contain both target classes.")
    return features, labels


def _collect_calibration_scores(
    paths: list[Path], model: XGBClassifier
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for features, target in _iter_patient_batches(
        paths, negative_sample_rate=None
    ):
        scores.append(model.predict_proba(features)[:, 1])
        labels.append(target)
    if not labels or len(np.unique(np.concatenate(labels))) != 2:
        raise ValueError("Calibration data must contain both target classes.")
    return np.concatenate(scores), np.concatenate(labels)


def _fit_platt_scaler(
    raw_probability: np.ndarray, labels: np.ndarray
) -> LogisticRegression:
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    raw_log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    scaler = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    scaler.fit(raw_log_odds, labels)
    return scaler


def _calibrate(
    scaler: LogisticRegression, raw_probability: np.ndarray
) -> np.ndarray:
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    raw_log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return scaler.predict_proba(raw_log_odds)[:, 1]


def _evaluate(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "windows": int(len(labels)),
        "sepsis_label_prevalence": float(np.mean(labels)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "decision_threshold": threshold,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "balanced_accuracy_at_threshold": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def _youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities, drop_intermediate=False
    )
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("Could not select a finite threshold from validation data.")
    youden_j = true_positive_rate - false_positive_rate
    best_score = np.max(youden_j[finite])
    if best_score <= 0:
        raise ValueError(
            "Validation predictions do not discriminate between target classes."
        )
    best_indices = np.flatnonzero(finite & np.isclose(youden_j, best_score))
    threshold = float(thresholds[best_indices[0]])
    if not 0 < threshold < 1:
        raise ValueError(
            "Validation-selected threshold is outside the supported probability "
            "range (0, 1)."
        )
    return threshold


def train_model(
    data_dir: Path,
    output_path: Path,
    *,
    train_split: str = "training_setA",
    test_split: str = "training_setB",
    validation_fraction: float = 0.2,
    negative_sample_rate: float = NEGATIVE_SAMPLE_RATE,
    threshold: float | None = None,
) -> dict[str, Any]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    if not 0 < negative_sample_rate <= 1:
        raise ValueError("negative_sample_rate must be in (0, 1].")
    if threshold is not None and not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one.")

    training_dir = data_dir / train_split
    test_dir = data_dir / test_split
    train_files = sorted(training_dir.glob("*.psv"))
    test_files = sorted(test_dir.glob("*.psv"))
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Expected patient PSV files in {training_dir} and {test_dir}."
        )
    if training_dir.resolve() == test_dir.resolve():
        raise ValueError("Training and held-out test splits must be different.")
    train_ids = {path.stem for path in train_files}
    overlapping_ids = sorted(train_ids.intersection(path.stem for path in test_files))
    if overlapping_ids:
        raise ValueError(
            "Patient IDs occur in both training and held-out test splits: "
            + ", ".join(overlapping_ids[:5])
        )

    development_files, validation_files = train_test_split(
        train_files,
        test_size=validation_fraction,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_x, train_y = _collect_training_data(
        development_files, negative_sample_rate
    )
    if len(np.unique(train_y)) != 2:
        raise ValueError(
            "The patient-level development split must contain both labels."
        )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=float((train_y == 0).sum() / (train_y == 1).sum()),
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(train_x, train_y)

    validation_raw, validation_y = _collect_calibration_scores(
        validation_files, model
    )
    calibrator = _fit_platt_scaler(validation_raw, validation_y)
    validation_calibrated = _calibrate(calibrator, validation_raw)
    threshold_method = "validation_youden_j"
    if threshold is None:
        threshold = _youden_threshold(validation_y, validation_calibrated)
    else:
        threshold_method = "user_supplied"
    validation_metrics = _evaluate(
        validation_y, validation_calibrated, threshold
    )

    test_raw, test_y = _collect_calibration_scores(test_files, model)
    test_calibrated = _calibrate(calibrator, test_raw)
    test_metrics = _evaluate(test_y, test_calibrated, threshold)

    metadata = {
        "feature_version": FEATURE_VERSION,
        "window_hours": WINDOW_HOURS,
        "prediction_target": (
            "First positive SepsisLabel occurs within the next "
            f"{FORECAST_HORIZON_HOURS} hours."
        ),
        "prediction_task": PREDICTION_TASK,
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "feature_names": list(FEATURE_NAMES),
        "training_split": train_split,
        "test_split": test_split,
        "patient_files": {
            "development": len(development_files),
            "validation": len(validation_files),
            "held_out_test": len(test_files),
        },
        "development_windows_after_negative_sampling": int(len(train_y)),
        "negative_sample_rate": negative_sample_rate,
        "random_seed": RANDOM_SEED,
        "library_versions": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")
        },
        "prediction_threshold": threshold,
        "threshold_method": threshold_method,
        "calibration": "Platt scaling fitted on patient-held-out development data",
        "validation_metrics": validation_metrics,
        "held_out_test_metrics": test_metrics,
    }
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def rethreshold_existing_model(
    data_dir: Path,
    model_path: Path,
    *,
    train_split: str = "training_setA",
    test_split: str = "training_setB",
) -> dict[str, Any]:
    """Select a validation-only threshold and refresh metrics without refitting."""
    from sepsispulse.inference import load_model_bundle

    training_dir = data_dir / train_split
    test_dir = data_dir / test_split
    train_files = sorted(training_dir.glob("*.psv"))
    test_files = sorted(test_dir.glob("*.psv"))
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Expected patient PSV files in {training_dir} and {test_dir}."
        )
    if training_dir.resolve() == test_dir.resolve():
        raise ValueError("Training and held-out test splits must be different.")
    train_ids = {path.stem for path in train_files}
    overlapping_ids = sorted(train_ids.intersection(path.stem for path in test_files))
    if overlapping_ids:
        raise ValueError(
            "Patient IDs occur in both training and held-out test splits: "
            + ", ".join(overlapping_ids[:5])
        )

    bundle = load_model_bundle(model_path)
    metadata = bundle["metadata"]
    if (
        metadata.get("training_split") != train_split
        or metadata.get("test_split") != test_split
    ):
        raise ValueError(
            "The model artifact was not trained with the requested dataset splits."
        )
    validation_fraction = 0.2
    _, validation_files = train_test_split(
        train_files,
        test_size=validation_fraction,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    model: XGBClassifier = bundle["model"]
    calibrator: LogisticRegression = bundle["calibrator"]
    validation_raw, validation_y = _collect_calibration_scores(
        validation_files, model
    )
    validation_calibrated = _calibrate(calibrator, validation_raw)
    threshold = _youden_threshold(validation_y, validation_calibrated)

    test_raw, test_y = _collect_calibration_scores(test_files, model)
    test_calibrated = _calibrate(calibrator, test_raw)
    updated_metadata = {
        **metadata,
        "prediction_target": (
            "First positive SepsisLabel occurs within the next "
            f"{FORECAST_HORIZON_HOURS} hours."
        ),
        "prediction_task": PREDICTION_TASK,
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "library_versions": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")
        },
        "validation_metrics": _evaluate(
            validation_y, validation_calibrated, threshold
        ),
        "held_out_test_metrics": _evaluate(test_y, test_calibrated, threshold),
        "prediction_threshold": threshold,
        "threshold_method": "validation_youden_j",
    }
    bundle["metadata"] = updated_metadata

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=model_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(bundle, temporary_path)
        temporary_path.replace(model_path)
        temporary_path = None
        model_path.with_suffix(".metrics.json").write_text(
            json.dumps(updated_metadata, indent=2), encoding="utf-8"
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return updated_metadata


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate the SepsisPulse XGBoost pipeline."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("physionet_data"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sepsispulse.joblib")
    )
    parser.add_argument("--train-split", default="training_setA")
    parser.add_argument("--test-split", default="training_setB")
    parser.add_argument("--negative-sample-rate", type=float, default=0.2)
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Override the default threshold selected by Youden's J on the "
            "patient-held-out validation set."
        ),
    )
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help=(
            "Recompute the validation-selected threshold and held-out metrics "
            "for an existing model without fitting XGBoost again."
        ),
    )
    args = parser.parse_args()
    if args.reuse_model:
        if args.threshold is not None:
            parser.error("--threshold cannot be used with --reuse-model")
        metadata = rethreshold_existing_model(
            args.data_dir,
            args.output,
            train_split=args.train_split,
            test_split=args.test_split,
        )
    else:
        metadata = train_model(
            args.data_dir,
            args.output,
            train_split=args.train_split,
            test_split=args.test_split,
            negative_sample_rate=args.negative_sample_rate,
            threshold=args.threshold,
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
