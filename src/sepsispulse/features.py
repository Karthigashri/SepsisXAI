from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from sepsispulse.schema import (
    CORE_VITAL_COLUMNS,
    FEATURE_SUFFIXES,
    MODEL_COLUMNS,
    STATIC_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    VITAL_COLUMNS,
    WINDOW_HOURS,
)

DYNAMIC_COLUMNS = tuple(
    column for column in MODEL_COLUMNS if column not in STATIC_COLUMNS
)
FEATURE_NAMES = tuple(
    f"{column}_{suffix}"
    for column in DYNAMIC_COLUMNS
    for suffix in FEATURE_SUFFIXES
) + STATIC_COLUMNS


class PatientDataError(ValueError):
    """Raised when a patient record cannot form a valid model input."""


def normalize_patient_frame(
    frame: pd.DataFrame, *, require_target: bool = False
) -> pd.DataFrame:
    if frame.empty:
        raise PatientDataError("Patient data is empty.")
    if TIME_COLUMN not in frame.columns:
        raise PatientDataError(f"Patient data must contain {TIME_COLUMN}.")

    allowed = set(MODEL_COLUMNS) | {TIME_COLUMN, TARGET_COLUMN}
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        raise PatientDataError(
            "Unsupported columns found: " + ", ".join(unexpected)
        )

    normalized = frame.copy()
    if require_target and TARGET_COLUMN not in normalized.columns:
        raise PatientDataError(
            f"Training data must contain {TARGET_COLUMN}."
        )
    for column in (*MODEL_COLUMNS, TARGET_COLUMN):
        if column not in normalized.columns:
            normalized[column] = np.nan
        if not pd.api.types.is_numeric_dtype(normalized[column].dtype):
            original = normalized[column]
            numeric = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & numeric.isna()
            if invalid.any():
                values = original[invalid].astype(str).unique()[:3].tolist()
                raise PatientDataError(
                    f"Column {column} contains non-numeric values: {values}"
                )
            normalized[column] = numeric

    numeric_values = normalized.loc[
        :, (*MODEL_COLUMNS, TARGET_COLUMN)
    ].to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(numeric_values).any():
        raise PatientDataError("Patient measurements contain infinite values.")

    times = pd.to_numeric(normalized[TIME_COLUMN], errors="coerce")
    if times.isna().any() or not np.isfinite(times.to_numpy()).all():
        raise PatientDataError(
            f"{TIME_COLUMN} must contain finite numeric ICU hours."
        )
    normalized[TIME_COLUMN] = times
    normalized = normalized.sort_values(TIME_COLUMN, kind="stable").reset_index(
        drop=True
    )
    time_differences = np.diff(normalized[TIME_COLUMN].to_numpy())
    if (time_differences <= 0).any():
        raise PatientDataError(
            f"{TIME_COLUMN} values must be unique and increasing."
        )
    if not np.isclose(time_differences, 1.0).all():
        raise PatientDataError(
            "Patient rows must cover consecutive one-hour ICU intervals; "
            "missing hours cannot be treated as observed hours."
        )

    if TARGET_COLUMN in normalized:
        if require_target and normalized[TARGET_COLUMN].isna().any():
            raise PatientDataError(
                f"{TARGET_COLUMN} cannot contain missing labels."
            )
        labels = normalized[TARGET_COLUMN].dropna().unique()
        if not set(labels).issubset({0.0, 1.0}):
            raise PatientDataError(f"{TARGET_COLUMN} must contain only 0 or 1.")
    return normalized


def _window_feature_matrix(values: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=WINDOW_HOURS, axis=0
    )
    windows = np.moveaxis(windows, -1, 1)
    observed = np.isfinite(windows)
    count = observed.sum(axis=1).astype(np.float64)
    clean = np.where(observed, windows, 0.0)

    mean = np.divide(
        clean.sum(axis=1),
        count,
        out=np.full(count.shape, np.nan),
        where=count > 0,
    )
    minimum = np.where(observed, windows, np.inf).min(axis=1)
    maximum = np.where(observed, windows, -np.inf).max(axis=1)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan

    positions = np.broadcast_to(
        np.arange(WINDOW_HOURS)[None, :, None], windows.shape
    )
    last_position = np.where(observed, positions, -1).max(axis=1)
    last_value = np.take_along_axis(
        windows,
        np.clip(last_position, 0, WINDOW_HOURS - 1)[:, None, :],
        axis=1,
    )[:, 0, :]
    last_value[last_position == -1] = np.nan

    time_values = np.where(observed, positions, 0.0)
    sum_t = time_values.sum(axis=1)
    sum_y = clean.sum(axis=1)
    sum_ty = (time_values * clean).sum(axis=1)
    sum_t_squared = (time_values * time_values * observed).sum(axis=1)
    denominator = count * sum_t_squared - sum_t * sum_t
    trend = np.divide(
        count * sum_ty - sum_t * sum_y,
        denominator,
        out=np.full(count.shape, np.nan),
        where=(count >= 2) & (denominator > 0),
    )

    return np.stack(
        (mean, minimum, maximum, last_value, trend, count), axis=-1
    ).reshape(len(windows), -1)


def _engineer_normalized_windows(normalized: pd.DataFrame) -> pd.DataFrame:
    if len(normalized) < WINDOW_HOURS:
        raise PatientDataError(
            f"At least {WINDOW_HOURS} hourly records are required; "
            f"received {len(normalized)}."
        )

    values = normalized.loc[:, DYNAMIC_COLUMNS].to_numpy(dtype=np.float64)
    dynamic_features = _window_feature_matrix(values)
    static_start = WINDOW_HOURS - 1
    static_features = normalized.loc[
        static_start:, STATIC_COLUMNS
    ].to_numpy(dtype=np.float64)
    matrix = np.column_stack((dynamic_features, static_features)).astype(
        np.float32
    )
    return pd.DataFrame(
        matrix,
        columns=FEATURE_NAMES,
        index=normalized[TIME_COLUMN].iloc[static_start:].to_numpy(),
    )


def _labels_for_normalized_windows(normalized: pd.DataFrame) -> np.ndarray:
    if len(normalized) < WINDOW_HOURS:
        return np.empty(0, dtype=np.int8)
    return (
        normalized[TARGET_COLUMN]
        .iloc[WINDOW_HOURS - 1 :]
        .to_numpy(dtype=np.int8)
    )


def engineer_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one feature row per complete trailing six-hour window."""
    normalized = normalize_patient_frame(frame)
    return _engineer_normalized_windows(normalized)


def labels_for_windows(frame: pd.DataFrame) -> np.ndarray:
    normalized = normalize_patient_frame(frame, require_target=True)
    return _labels_for_normalized_windows(normalized)


def engineer_labeled_windows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Normalize a labeled patient once, then align features and endpoint labels."""
    normalized = normalize_patient_frame(frame, require_target=True)
    return (
        _engineer_normalized_windows(normalized),
        _labels_for_normalized_windows(normalized),
    )


def forecast_labels_for_windows(
    frame: pd.DataFrame, *, horizon_hours: int
) -> np.ndarray:
    """Label future onset within a horizon; -1 marks unusable/censored windows."""
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be at least one.")
    normalized = normalize_patient_frame(frame, require_target=True)
    return _forecast_labels_for_normalized_windows(normalized, horizon_hours)


def _forecast_labels_for_normalized_windows(
    normalized: pd.DataFrame, horizon_hours: int
) -> np.ndarray:
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be at least one.")
    if len(normalized) < WINDOW_HOURS:
        return np.empty(0, dtype=np.int8)
    labels = normalized[TARGET_COLUMN].to_numpy(dtype=np.int8)
    positive_indices = np.flatnonzero(labels == 1)
    onset_index = int(positive_indices[0]) if len(positive_indices) else None
    window_end_indices = np.arange(WINDOW_HOURS - 1, len(normalized))
    outcomes = np.full(len(window_end_indices), -1, dtype=np.int8)
    last_observed_index = len(normalized) - 1

    for output_index, current_index in enumerate(window_end_indices):
        if onset_index is not None and current_index >= onset_index:
            continue
        if onset_index is not None and onset_index <= current_index + horizon_hours:
            outcomes[output_index] = 1
        elif current_index + horizon_hours <= last_observed_index:
            outcomes[output_index] = 0

    return outcomes


def engineer_forecast_windows(
    frame: pd.DataFrame, *, horizon_hours: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build six-hour inputs and future-onset targets from a patient series."""
    normalized = normalize_patient_frame(frame, require_target=True)
    features = _engineer_normalized_windows(normalized)
    labels = _forecast_labels_for_normalized_windows(normalized, horizon_hours)
    if len(labels) != len(features):
        raise PatientDataError("Forecast labels did not align with input windows.")
    return features, labels


def core_vital_coverage(frame: pd.DataFrame) -> float:
    normalized = normalize_patient_frame(frame)
    window = normalized.tail(WINDOW_HOURS)
    available = sum(
        column in window and window[column].notna().any()
        for column in CORE_VITAL_COLUMNS
    )
    return float(available / len(CORE_VITAL_COLUMNS))


def validate_feature_names(feature_names: Iterable[str]) -> None:
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError(
            "Model feature schema does not match this application version."
        )
