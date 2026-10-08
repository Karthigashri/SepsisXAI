"""Observed column contract for the PhysioNet/CinC 2019 PSV files."""

TARGET_COLUMN = "SepsisLabel"
TIME_COLUMN = "ICULOS"

VITAL_COLUMNS = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
)

LAB_COLUMNS = (
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
)

STATIC_COLUMNS = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")

MODEL_COLUMNS = VITAL_COLUMNS + LAB_COLUMNS + STATIC_COLUMNS
CORE_VITAL_COLUMNS = ("HR", "O2Sat", "Temp", "SBP", "MAP", "Resp")

FEATURE_SUFFIXES = (
    "mean_6h",
    "min_6h",
    "max_6h",
    "last_6h",
    "trend_6h",
    "count_6h",
)

WINDOW_HOURS = 6
FEATURE_VERSION = "six-hour-summary-v1"
FORECAST_HORIZON_HOURS = 6
PREDICTION_TASK = "first_sepsis_label_within_forecast_horizon"
