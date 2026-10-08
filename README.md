# SepsisPulse

SepsisPulse is an academic research prototype for confidence-aware, explainable
sepsis risk assessment from ICU time-series data. It is not a medical device,
does not diagnose sepsis, and must not replace clinician judgement.

> **Research/Clinical Decision-Support Prototype — Not for Medical Diagnosis.**

## Dataset inspection

The current local data was inspected across every file before feature
engineering:

- `training_setA`: 20,336 patient files, 790,215 rows, 1,790 patients with a
  positive label, and 17,136 positive hourly labels.
- `training_setB`: 20,000 patient files, 761,995 rows, 1,142 patients with a
  positive label, and 10,780 positive hourly labels.
- The complete dataset has 40,336 files and 1,552,210 rows. Files contain 41
  consistently ordered numeric columns, hourly observations, and no malformed
  rows. `SepsisLabel` is the target; `ICULOS` is the ICU-hour index rather than
  a calendar timestamp. Patient IDs are the PSV file names.
- Missingness is substantial and differs between splits. For example, in set
  A the missing rates are 7.74% for heart rate, 66.22% for temperature, 92.49%
  for WBC, and 96.57% for lactate. `EtCO2` is fully missing in set A but is
  present in some set B rows. The model therefore retains missingness rather
  than assuming imputed clinical observations.

Run the repeatable, memory-bounded audit from the repository root:

```powershell
python -m pip install -e ".[dev]"
python scripts/inspect_dataset.py --data-dir physionet_data
python scripts/inspect_dataset.py --data-dir physionet_data --format json
```

The audit reports file names, patient IDs, per-split row/label counts, inferred
column types and roles, per-column missingness, and `ICULOS` increments. Use
`--limit 10` for a faster schema smoke check; a complete scan reads every row.

The data directory is intentionally ignored by Git. Obtain and use the dataset
under the terms of its source; do not commit patient-level data or generated
model artifacts.

## Modeling approach

The initial pipeline creates trailing six-hour features for each hourly
decision: mean, minimum, maximum, latest observed value, linear trend, and
measurement count for each vital/lab variable, plus the observed static fields.
Rows require six consecutive ICU-hour records. XGBoost is trained on set A,
with a patient-level development/validation split. Negative development
windows are subsampled for tractable fitting; Platt scaling is fitted on the
unsubsampled patient-held-out validation windows. Set B remains held out for
evaluation. Metrics are window-level and must not be interpreted as clinical
performance. The target for an input window ending at ICU hour `t` is whether
the patient's **first positive `SepsisLabel` occurs in one of the next six
observed ICU hours**. Rows at or after the first positive label are excluded.
Negative examples are used only when a full six-hour follow-up is present;
right-censored windows are excluded rather than incorrectly labeled negative.
The first positive `SepsisLabel` is only a dataset-label proxy; consult the
dataset's official documentation before interpreting it as clinical onset.
The default high/low cutoff is selected with Youden's J using only the
patient-held-out validation scores, then locked for set B evaluation. It is an
academic operating point, not a recommended clinical threshold.

### Initial six-hour forecast evaluation

The first run of the future-label pipeline on held-out set B selected a
validation-only threshold of 0.01204. Set B had 539,795 eligible windows and
0.872% positive-window prevalence. Results were AUROC 0.595, average precision
0.0140, sensitivity 0.258, specificity 0.865, and precision 0.0166. This is
weak discrimination and poor precision; the model is not suitable for
clinical use. These results are window-level, use a single patient-separated
dataset split, and are not prospective or external validation. Detailed
metrics are written to the local ignored
`artifacts/sepsispulse.metrics.json`.

The earlier contemporaneous-label artifact is not valid evidence for this
forecasting task; model loading now checks the task and horizon to prevent
confusing the two.

The dashboard's confidence level is explicitly a **heuristic**, combining
the margin from the risk cutoff (normalized separately on each side of the
cutoff) with coverage of six core vital signs. It is not a probability that a
prediction is correct. SHAP values
explain the base XGBoost model on its log-odds scale; the risk shown to the
user is then Platt-calibrated. Neither the confidence nor explanations have
been validated for clinical use.

Trees do not require feature normalization. Values are checked for malformed
and infinite measurements, but extreme finite values are preserved because
blindly clipping physiological outliers can erase clinically meaningful
signals.

## Train and run locally

```powershell
python -m pip install -e ".[dev]"
python -m sepsispulse.training --data-dir physionet_data --output artifacts/sepsispulse.joblib
uvicorn sepsispulse.api:app --reload
```

Use `python -m sepsispulse.training --data-dir physionet_data --output artifacts/sepsispulse.joblib --reuse-model` only after training a model for this same prediction task; this reselects the validation threshold and refreshes held-out metrics without refitting XGBoost.

Open `http://127.0.0.1:8000`. The API serves the dashboard, accepts a PSV
upload or JSON hourly records, and returns calibrated risk of a first positive
dataset label within six hours, heuristic confidence, trends, missingness, and
SHAP contributors. It rejects labeled patient data that already contains a
positive label instead of presenting that case as a pre-onset forecast. A trained model
artifact is required; without one the dashboard clearly reports that
predictions are unavailable.

## Deploy on Render

The repository includes a Render Blueprint. A trained artifact is deliberately
not committed, so deployment requires a trusted HTTPS artifact URL and its
SHA-256 digest:

1. Train the model as above and upload `artifacts/sepsispulse.joblib` to
   artifact storage that Render can reach.
2. Compute its digest with
   `Get-FileHash artifacts\sepsispulse.joblib -Algorithm SHA256`.
3. Create a Render Blueprint from this repository and provide
   `SEPSISPULSE_MODEL_URL` and `SEPSISPULSE_MODEL_SHA256` as secret environment
   values. The build downloads the artifact over HTTPS, verifies the digest,
   validates the bundle, and only then starts the API.

Use an artifact URL and hash that you control; the bundle uses Python
serialization and must never be loaded from an untrusted source. Do not put
signed URLs, credentials, patient data, or secrets in the repository. A live
deployment also requires the project owner’s hosting account and artifact
storage access. This demonstration API has no authentication and is not
approved for protected health information; deploy it only for non-clinical,
de-identified academic demonstrations. In-memory processing does not protect
data in transit or make a public deployment appropriate for identifiable
patient data.

## Safety and scope

**Research/Clinical Decision-Support Prototype — Not for Medical Diagnosis.**
This project is not a medical device, does not diagnose or rule out sepsis,
and must not replace a qualified clinician. Risk thresholds and model outputs
are research settings only. Patient-held-out evaluation on this dataset does
not establish external validity, prospective utility, or clinical safety.
