#!/usr/bin/env python3
"""Scorer for materials feature-engineering tasks.

The contestant submits two CSVs of derived features keyed by `id`:

  - train_features.csv: feature vectors for every training row.
  - test_features.csv:  feature vectors for every test row.

Both files must share the same set of feature columns. The scorer fits a
RandomForestRegressor (sklearn defaults, n_estimators=100) on the train
features and the hidden train labels, predicts on the test features, and
reports R^2 (or R^2 in log10 space, for targets that span many orders of
magnitude) averaged across 5 fixed seeds. The primary `score` field is
that mean R^2 clipped to [0, 1].
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
except Exception as exc:  # pragma: no cover - reported at runtime
    np = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_labels(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "target"]:
            raise ValueError(f"{path.name} must have columns id,target")
        labels: dict[str, float] = {}
        for row in reader:
            row_id = row["id"].strip()
            if not row_id:
                raise ValueError(f"{path.name} contains an empty id")
            if row_id in labels:
                raise ValueError(f"duplicate id in {path.name}: {row_id}")
            labels[row_id] = float(row["target"])
    if not labels:
        raise ValueError(f"{path.name} contains no rows")
    return labels


def read_features(path: Path, expected_ids: set[str]) -> tuple[list[str], dict[str, list[float]]]:
    if not path.is_file():
        raise ValueError(f"{path.name} was not found")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path.name} is empty") from exc
        if not header or header[0] != "id":
            raise ValueError(f"{path.name} must start with an 'id' column")
        feature_columns = header[1:]
        if not feature_columns:
            raise ValueError(f"{path.name} must contain at least one feature column besides id")
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError(f"{path.name} has duplicate feature column names")

        rows: dict[str, list[float]] = {}
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"{path.name} row has {len(row)} fields, expected {len(header)}")
            row_id = row[0].strip()
            if not row_id:
                raise ValueError(f"{path.name} contains an empty id")
            if row_id in rows:
                raise ValueError(f"duplicate id in {path.name}: {row_id}")
            values: list[float] = []
            for col_name, cell in zip(feature_columns, row[1:]):
                try:
                    val = float(cell)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path.name} {row_id!r} column {col_name!r} is not numeric: {cell!r}"
                    ) from exc
                if not math.isfinite(val):
                    raise ValueError(
                        f"{path.name} {row_id!r} column {col_name!r} is not finite: {val}"
                    )
                values.append(val)
            rows[row_id] = values

    submitted = set(rows)
    missing = sorted(expected_ids - submitted)
    if missing:
        raise ValueError(f"{path.name} is missing {len(missing)} ids; first missing id: {missing[0]}")
    extra = sorted(submitted - expected_ids)
    if extra:
        raise ValueError(f"{path.name} contains {len(extra)} unexpected ids; first unexpected id: {extra[0]}")
    return feature_columns, rows


def stack(ids: list[str], rows: dict[str, list[float]]):
    return np.asarray([rows[rid] for rid in ids], dtype=np.float64)


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def fit_and_score(
    x_train,
    y_train_fit,
    x_test,
    y_test_score,
    seeds: list[int],
) -> tuple[float, list[float]]:
    per_seed: list[float] = []
    for seed in seeds:
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
        model.fit(x_train, y_train_fit)
        preds = model.predict(x_test)
        per_seed.append(r2(y_test_score, preds))
    return float(np.mean(per_seed)), per_seed


def normalized_score(value: float, metadata: dict[str, object]) -> tuple[float, float]:
    baseline = float(metadata["baseline"])
    ideal = float(metadata["ideal"])
    direction = str(metadata.get("direction", "higher"))
    if direction == "higher":
        improvement = (value - baseline) / max(1e-12, ideal - baseline)
    elif direction == "lower":
        improvement = (baseline - value) / max(1e-12, baseline - ideal)
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    return max(0.0, min(1.0, improvement)), improvement


def main() -> int:
    pred_dir = Path(sys.argv[1])
    ref_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {"score": 0.0, "metric_value": 0.0, "improvement": 0.0}
    try:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                f"scorer needs numpy and scikit-learn but failed to import: {_IMPORT_ERROR}"
            )
        metadata = json.loads((ref_dir / "metadata.json").read_text(encoding="utf-8"))
        metric = str(metadata.get("metric", "r2"))
        seeds = [int(s) for s in metadata.get("rf_seeds", [0, 1, 2, 3, 4])]

        train_labels = read_labels(ref_dir / "train_labels.csv")
        test_labels = read_labels(ref_dir / "answers.csv")

        train_cols, train_rows = read_features(pred_dir / "train_features.csv", set(train_labels))
        test_cols, test_rows = read_features(pred_dir / "test_features.csv", set(test_labels))
        if train_cols != test_cols:
            raise ValueError(
                "train_features.csv and test_features.csv must share the same feature columns "
                "in the same order"
            )

        train_ids = sorted(train_labels)
        test_ids = sorted(test_labels)
        x_train = stack(train_ids, train_rows)
        x_test = stack(test_ids, test_rows)
        y_train = np.asarray([train_labels[rid] for rid in train_ids], dtype=np.float64)
        y_test = np.asarray([test_labels[rid] for rid in test_ids], dtype=np.float64)

        if metric == "r2":
            mean_r2, per_seed = fit_and_score(x_train, y_train, x_test, y_test, seeds)
        elif metric == "r2_log10":
            floor = float(metadata.get("log10_clip_floor", 1e-20))
            y_train_log = np.log10(np.clip(y_train, floor, None))
            y_test_log = np.log10(np.clip(y_test, floor, None))
            mean_r2, per_seed = fit_and_score(x_train, y_train_log, x_test, y_test_log, seeds)
        else:
            raise ValueError(f"unknown metric: {metric!r}")

        score, improvement = normalized_score(mean_r2, metadata)
        scores = {
            "score": score,
            "metric_value": mean_r2,
            "improvement": improvement,
            "n_features": float(len(train_cols)),
            "n_train": float(len(train_ids)),
            "n_test": float(len(test_ids)),
        }
        scores[metric] = mean_r2
        scores["per_seed_r2"] = per_seed  # type: ignore[assignment]
        for k in ("baseline", "ideal", "rf_reference_score"):
            if k in metadata:
                scores[k] = float(metadata[k])
    except Exception as exc:
        (out_dir / "scoring_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    write_json(out_dir / "scores.json", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
