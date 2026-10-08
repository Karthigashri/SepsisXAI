import numpy as np
import pandas as pd
import pytest

from sepsispulse.features import (
    FEATURE_NAMES,
    core_vital_coverage,
    engineer_windows,
    engineer_forecast_windows,
    forecast_labels_for_windows,
    labels_for_windows,
    normalize_patient_frame,
)
from sepsispulse.inference import _confidence_score, predict_patient
from sepsispulse.schema import TARGET_COLUMN, TIME_COLUMN, WINDOW_HOURS


def _synthetic_frame(hours: int = 7) -> pd.DataFrame:
    return pd.DataFrame(
        {
            TIME_COLUMN: np.arange(1, hours + 1),
            "HR": np.arange(60, 60 + hours, dtype=float),
            "O2Sat": [97.0] * hours,
            "Temp": [np.nan] * hours,
            "SepsisLabel": [0] * (hours - 1) + [1],
        }
    )


def _forecast_frame(hours: int, onset_index: int | None) -> pd.DataFrame:
    labels = np.zeros(hours, dtype=int)
    if onset_index is not None:
        labels[onset_index:] = 1
    return pd.DataFrame(
        {
            TIME_COLUMN: np.arange(1, hours + 1),
            "HR": np.arange(60, 60 + hours, dtype=float),
            "SepsisLabel": labels,
        }
    )


def test_engineer_windows_summarizes_six_hour_history():
    features = engineer_windows(_synthetic_frame())

    assert features.shape == (2, len(FEATURE_NAMES))
    assert features.iloc[0]["HR_mean_6h"] == pytest.approx(62.5)
    assert features.iloc[0]["HR_last_6h"] == pytest.approx(65)
    assert features.iloc[0]["HR_trend_6h"] == pytest.approx(1)
    assert features.iloc[0]["HR_count_6h"] == 6
    assert np.isnan(features.iloc[0]["Temp_mean_6h"])
    assert features.iloc[0]["Temp_count_6h"] == 0


def test_labels_align_to_end_of_each_complete_window():
    assert labels_for_windows(_synthetic_frame()).tolist() == [0, 1]


def test_short_or_nonconsecutive_windows_are_rejected():
    with pytest.raises(ValueError, match="At least 6"):
        engineer_windows(_synthetic_frame(hours=WINDOW_HOURS - 1))

    frame = _synthetic_frame()
    frame.loc[3, TIME_COLUMN] = 10
    with pytest.raises(ValueError, match="consecutive"):
        normalize_patient_frame(frame)


def test_non_numeric_measurements_and_labels_are_rejected():
    frame = _synthetic_frame()
    frame["HR"] = frame["HR"].astype(object)
    frame.loc[0, "HR"] = "unavailable"
    with pytest.raises(ValueError, match="non-numeric"):
        normalize_patient_frame(frame)

    frame = _synthetic_frame()
    frame.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="only 0 or 1"):
        normalize_patient_frame(frame, require_target=True)


def test_core_vital_coverage_uses_available_fields_in_last_window():
    frame = _synthetic_frame()
    assert core_vital_coverage(frame) == pytest.approx(2 / 6)


def test_forecast_labels_identify_onset_in_next_six_hours_only():
    features, labels = engineer_forecast_windows(
        _forecast_frame(15, onset_index=8), horizon_hours=6
    )

    assert len(features) == len(labels) == 10
    assert labels.tolist() == [1, 1, 1, -1, -1, -1, -1, -1, -1, -1]


def test_forecast_targets_separate_future_onset_from_known_negative_followup():
    labels = forecast_labels_for_windows(
        _forecast_frame(15, onset_index=12), horizon_hours=6
    )

    assert labels.tolist() == [0, 1, 1, 1, 1, 1, 1, -1, -1, -1]


def test_no_onset_negative_windows_require_complete_followup():
    labels = forecast_labels_for_windows(
        _forecast_frame(15, onset_index=None), horizon_hours=6
    )

    assert labels.tolist() == [0, 0, 0, 0, -1, -1, -1, -1, -1, -1]


def test_forecast_horizon_must_be_positive():
    with pytest.raises(ValueError, match="at least one"):
        forecast_labels_for_windows(
            _forecast_frame(15, onset_index=8), horizon_hours=0
        )


def test_forecast_inference_rejects_records_after_positive_label():
    frame = _forecast_frame(6, onset_index=4)

    with pytest.raises(ValueError, match="already contain a positive label"):
        predict_patient({}, frame, "already-positive")


def test_confidence_measures_margin_on_the_correct_side_of_threshold():
    assert _confidence_score(0.01, 0.02, 1.0) == pytest.approx(0.5)
    assert _confidence_score(1.0, 0.02, 1.0) == pytest.approx(1.0)
    assert _confidence_score(0.02, 0.02, 1.0) == 0
    assert _confidence_score(0.01, 0.02, 0.5) == pytest.approx(0.25)
