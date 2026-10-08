from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sepsispulse.features import (
    FEATURE_NAMES,
    PatientDataError,
    core_vital_coverage,
    engineer_windows,
    normalize_patient_frame,
    validate_feature_names,
)
from sepsispulse.schema import (
    FEATURE_VERSION,
    FORECAST_HORIZON_HOURS,
    MODEL_COLUMNS,
    PREDICTION_TASK,
    TARGET_COLUMN,
    TIME_COLUMN,
    VITAL_COLUMNS,
    WINDOW_HOURS,
)


def load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Model artifact has an unsupported format.")
    if not {"model", "calibrator", "metadata"}.issubset(bundle):
        raise ValueError("Model artifact is missing required components.")
    metadata = bundle["metadata"]
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model artifact uses an unsupported feature version.")
    if (
        metadata.get("prediction_task") != PREDICTION_TASK
        or metadata.get("forecast_horizon_hours") != FORECAST_HORIZON_HOURS
    ):
        raise ValueError("Model artifact does not predict the configured future horizon.")
    validate_feature_names(metadata.get("feature_names", ()))
    if not isinstance(bundle["model"], XGBClassifier):
        raise ValueError("Model artifact does not contain an XGBoost classifier.")
    if not isinstance(bundle["calibrator"], LogisticRegression):
        raise ValueError("Model artifact does not contain a Platt calibrator.")
    return bundle


def _calibrated_probability(
    calibrator: LogisticRegression, raw_probability: float
) -> float:
    clipped = float(np.clip(raw_probability, 1e-6, 1 - 1e-6))
    raw_log_odds = np.array([[np.log(clipped / (1 - clipped))]])
    return float(calibrator.predict_proba(raw_log_odds)[0, 1])


def _confidence_score(
    probability: float, threshold: float, vital_coverage: float
) -> float:
    margin_scale = 1 - threshold if probability >= threshold else threshold
    threshold_margin = abs(probability - threshold) / margin_scale
    return float(np.clip(threshold_margin * vital_coverage, 0, 1))


def _shap_contributions(
    model: XGBClassifier, feature_values: np.ndarray
) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(feature_values)
    if isinstance(raw_values, list):
        if len(raw_values) != 2:
            raise ValueError("SHAP returned an unexpected class output.")
        raw_values = raw_values[1]
    contribution_values = np.asarray(raw_values)
    if contribution_values.ndim == 3:
        if contribution_values.shape[0] == 1:
            contribution_values = contribution_values[0, :, -1]
        elif contribution_values.shape[1] == 1:
            contribution_values = contribution_values[-1, 0, :]
        else:
            raise ValueError(
                "SHAP returned an unexpected contribution shape: "
                f"{contribution_values.shape}"
            )
    elif contribution_values.ndim == 2:
        contribution_values = contribution_values[0]
    if contribution_values.shape != (len(FEATURE_NAMES),):
        raise ValueError(
            "SHAP returned an unexpected contribution shape: "
            f"{contribution_values.shape}"
        )
    return contribution_values


def predict_patient(
    bundle: dict[str, Any], frame: pd.DataFrame, patient_id: str
) -> dict[str, Any]:
    normalized = normalize_patient_frame(frame)
    if len(normalized) < WINDOW_HOURS:
        raise PatientDataError(
            f"At least {WINDOW_HOURS} hourly records are required; "
            f"received {len(normalized)}."
        )
    if normalized[TARGET_COLUMN].eq(1).any():
        raise PatientDataError(
            "This model forecasts a first positive sepsis label. The supplied "
            "records already contain a positive label, so they are not a "
            "pre-onset forecast input."
        )
    feature_frame = engineer_windows(normalized).tail(1)
    feature_values = feature_frame.to_numpy(dtype=np.float32)
    model: XGBClassifier = bundle["model"]
    calibrator: LogisticRegression = bundle["calibrator"]
    metadata = bundle["metadata"]

    raw_probability = float(model.predict_proba(feature_values)[0, 1])
    probability = _calibrated_probability(calibrator, raw_probability)
    threshold = float(metadata["prediction_threshold"])
    risk_high = probability >= threshold

    coverage = core_vital_coverage(normalized)
    confidence_score = _confidence_score(probability, threshold, coverage)
    confidence_high = confidence_score >= 0.5
    if risk_high:
        alert = (
            "Review required — high-risk model output"
            if confidence_high
            else "Review required — high-risk output has low confidence"
        )
    elif confidence_high:
        alert = "No high-risk alert from the model"
    else:
        alert = "Interpret with caution — low-confidence model output"

    window = normalized.tail(WINDOW_HOURS)
    missingness = {
        column: round(float(window[column].isna().mean()), 4)
        for column in MODEL_COLUMNS
    }
    trends = {
        column: [
            None if pd.isna(value) else float(value)
            for value in window[column].tolist()
        ]
        for column in (
            *VITAL_COLUMNS,
            "WBC",
            "Lactate",
            "Platelets",
            "Creatinine",
        )
    }

    contributions = _shap_contributions(model, feature_values)
    most_influential = np.argsort(np.abs(contributions))[::-1][:8]
    explanations = [
        {
            "feature": FEATURE_NAMES[index],
            "value": (
                None
                if pd.isna(feature_values[0, index])
                else float(feature_values[0, index])
            ),
            "shap_value": float(contributions[index]),
        }
        for index in most_influential
    ]

    return {
        "patient_id": patient_id,
        "prediction_time_iculos": float(normalized[TIME_COLUMN].iloc[-1]),
        "window_hours": WINDOW_HOURS,
        "prediction_horizon_hours": FORECAST_HORIZON_HOURS,
        "prediction_target": metadata["prediction_target"],
        "risk_probability": probability,
        "risk_percent": round(100 * probability, 1),
        "risk_category": "high" if risk_high else "low",
        "confidence_level": "high" if confidence_high else "low",
        "confidence_score": round(confidence_score, 4),
        "confidence_method": (
            "Heuristic combining distance from the configured risk threshold "
            "and availability of six core vital signs; not a probability of "
            "prediction correctness."
        ),
        "core_vital_coverage": round(coverage, 4),
        "alert_status": alert,
        "threshold": threshold,
        "trends": trends,
        "missingness": missingness,
        "explanation_scale": (
            "SHAP contributions for the base XGBoost model (log-odds scale); "
            "the displayed probability is subsequently Platt-calibrated."
        ),
        "top_contributors": explanations,
        "model_status": "ready",
        "feature_version": FEATURE_VERSION,
    }
