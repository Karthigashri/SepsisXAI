import numpy as np
import pandas as pd
import pytest

from sepsispulse.training import _evaluate, _patient_windows, _youden_threshold


def test_youden_threshold_is_selected_from_validation_scores():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.01, 0.02, 0.1, 0.7, 0.8, 0.95])

    threshold = _youden_threshold(labels, probabilities)
    metrics = _evaluate(labels, probabilities, threshold)

    assert threshold == pytest.approx(0.7)
    assert metrics["sensitivity"] == 1
    assert metrics["specificity"] == 1
    assert metrics["precision"] == 1


def test_youden_threshold_rejects_degenerate_equal_probabilities():
    with pytest.raises(ValueError, match="do not discriminate"):
        _youden_threshold(
            np.array([0, 0, 1, 1]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        )


def test_patient_training_windows_exclude_censored_rows(tmp_path):
    labels = [0] * 12 + [1] * 3
    patient_file = tmp_path / "patient.psv"
    pd.DataFrame(
        {
            "ICULOS": np.arange(1, 16),
            "HR": np.arange(60, 75),
            "SepsisLabel": labels,
        }
    ).to_csv(patient_file, sep="|", index=False)

    features, outcomes = _patient_windows(patient_file)

    assert len(features) == len(outcomes) == 7
    assert outcomes.tolist() == [0, 1, 1, 1, 1, 1, 1]
