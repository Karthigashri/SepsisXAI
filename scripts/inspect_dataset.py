from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sepsispulse.schema import (
    LAB_COLUMNS,
    STATIC_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    VITAL_COLUMNS,
)

MISSING_MARKERS = {"", "nan", "na", "null"}


def _role(column: str) -> str:
    if column in VITAL_COLUMNS:
        return "vital sign"
    if column in LAB_COLUMNS:
        return "laboratory"
    if column in STATIC_COLUMNS:
        return "demographic/context"
    if column == TIME_COLUMN:
        return "hour index (ICULOS)"
    if column == TARGET_COLUMN:
        return "target label"
    return "unclassified"


def inspect_dataset(root: Path, limit: int | None = None) -> dict[str, Any]:
    """Inspect PSV files without loading the full dataset into memory."""
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    splits: dict[str, Any] = {}
    reference_header: list[str] | None = None
    files = sorted(root.glob("training_set*/*.psv"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No training_set*/.psv files found under {root}")

    by_split: dict[str, list[Path]] = {}
    for file_path in files:
        by_split.setdefault(file_path.parent.name, []).append(file_path)

    for split, split_files in sorted(by_split.items()):
        row_count = 0
        label_counts: Counter[str] = Counter()
        missing_counts: Counter[str] = Counter()
        observed_types: dict[str, set[str]] = {}
        row_lengths: list[int] = []
        positive_patients = 0
        time_steps: Counter[float] = Counter()
        malformed_rows = 0
        first_header: list[str] | None = None

        for file_path in split_files:
            patient_positive = False
            previous_iculos: float | None = None
            patient_rows = 0
            with file_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, delimiter="|")
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"Empty PSV file: {file_path}") from exc

                if first_header is None:
                    first_header = header
                if header != first_header:
                    raise ValueError(f"Inconsistent column order in {file_path}")
                if reference_header is None:
                    reference_header = header
                elif header != reference_header:
                    raise ValueError(
                        f"Schema differs from other split files: {file_path}"
                    )
                missing_required = {TARGET_COLUMN, TIME_COLUMN} - set(header)
                if missing_required:
                    raise ValueError(
                        f"{file_path} is missing required columns: "
                        + ", ".join(sorted(missing_required))
                    )
                indexes = {name: index for index, name in enumerate(header)}

                for row in reader:
                    row_count += 1
                    patient_rows += 1
                    if len(row) != len(header):
                        malformed_rows += 1
                        continue
                    for index, value in enumerate(row):
                        name = header[index]
                        value = value.strip()
                        if value.lower() in MISSING_MARKERS:
                            missing_counts[name] += 1
                        else:
                            try:
                                float(value)
                            except ValueError:
                                observed_types.setdefault(name, set()).add("text")
                            else:
                                observed_types.setdefault(name, set()).add(
                                    "numeric"
                                )

                    label = row[indexes[TARGET_COLUMN]]
                    label_counts[label] += 1
                    patient_positive |= label == "1"

                    try:
                        iculos = float(row[indexes[TIME_COLUMN]])
                    except (KeyError, ValueError):
                        continue
                    if previous_iculos is not None:
                        time_steps[iculos - previous_iculos] += 1
                    previous_iculos = iculos

            row_lengths.append(patient_rows)
            positive_patients += int(patient_positive)

        if first_header is None:
            continue
        time_fields = [
            name
            for name in first_header
            if name == TIME_COLUMN
            or name.lower() in {"timestamp", "datetime", "date", "time"}
        ]
        calendar_timestamps = [
            name for name in time_fields if name != TIME_COLUMN
        ]
        columns = []
        for name in first_header:
            count = observed_types.get(name, set())
            dtype = (
                "numeric (nullable)"
                if count == {"numeric"} or not count
                else "text/mixed"
                if count == {"text"}
                else "mixed"
            )
            columns.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "role": _role(name),
                    "missing_count": missing_counts[name],
                    "missing_percent": (
                        round(100 * missing_counts[name] / row_count, 2)
                        if row_count
                        else None
                    ),
                }
            )
        splits[split] = {
            "files": len(split_files),
            "sample_patient_ids": [path.stem for path in split_files[:5]],
            "sample_file_names": [path.name for path in split_files[:5]],
            "rows": row_count,
            "row_length_min": min(row_lengths) if row_lengths else 0,
            "row_length_max": max(row_lengths) if row_lengths else 0,
            "positive_patients": positive_patients,
            "label_rows": dict(sorted(label_counts.items())),
            "time_fields": time_fields,
            "calendar_timestamps_present": bool(calendar_timestamps),
            "malformed_rows": malformed_rows,
            "iculos_increments": {
                str(step): count for step, count in sorted(time_steps.items())
            },
            "columns": columns,
        }

    if reference_header is None:
        raise ValueError(f"No readable PSV data found under {root}")
    return {
        "dataset_root": str(root.resolve()),
        "files_scanned": sum(item["files"] for item in splits.values()),
        "complete_scan": limit is None,
        "splits": splits,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"Dataset: {report['dataset_root']}",
        f"Files scanned: {report['files_scanned']}",
        f"Complete scan: {report['complete_scan']}",
    ]
    for split, details in report["splits"].items():
        lines.extend(
            [
                "",
                f"## {split}",
                f"Files/patients: {details['files']}",
                f"Example file names: {', '.join(details['sample_file_names'])}",
                f"Rows: {details['rows']}",
                "Rows per patient (min-max): "
                f"{details['row_length_min']}-{details['row_length_max']}",
                f"Patients with SepsisLabel=1: {details['positive_patients']}",
                f"Label rows: {details['label_rows']}",
                f"Time fields: {details['time_fields']}",
                "Calendar timestamps present: "
                f"{details['calendar_timestamps_present']}",
                f"ICULOS increments (hours): {details['iculos_increments']}",
                f"Malformed rows: {details['malformed_rows']}",
                "",
                "| Column | Inferred dtype | Role | Missing | Missing % |",
                "|---|---|---|---:|---:|",
            ]
        )
        for column in details["columns"]:
            lines.append(
                f"| {column['name']} | {column['dtype']} | {column['role']} "
                f"| {column['missing_count']} | {column['missing_percent']}% |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PhysioNet PSV files without loading them all into memory."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("physionet_data"),
        help="Root containing training_setA and training_setB.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Inspect only the first N files for a quick structural check.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")

    report = inspect_dataset(args.data_dir, args.limit)
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report))


if __name__ == "__main__":
    main()
