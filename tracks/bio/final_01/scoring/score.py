#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path.name} is missing a header")
        return list(reader)


def read_predictions(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "prediction"]:
            raise ValueError("predictions.csv must contain exactly id,prediction columns")
        predictions: dict[str, float] = {}
        for row in reader:
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                raise ValueError("predictions.csv contains an empty id")
            if row_id in predictions:
                raise ValueError(f"duplicate prediction id: {row_id}")
            value = float(row["prediction"])
            if not math.isfinite(value):
                raise ValueError(f"prediction for {row_id} is not finite")
            predictions[row_id] = value
    if not predictions:
        raise ValueError("predictions.csv contains no rows")
    return predictions


def ensure_complete(predictions: dict[str, float], answers: list[dict[str, str]]) -> None:
    expected_ids = [row["id"] for row in answers]
    expected_set = set(expected_ids)
    missing = [row_id for row_id in expected_ids if row_id not in predictions]
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} ids, first: {missing[:3]}")
    extra = [row_id for row_id in predictions if row_id not in expected_set]
    if extra:
        raise ValueError(f"unexpected prediction ids: {extra[:3]}")


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    denom = denom_x * denom_y
    if denom == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearmanr(y_true: list[float], y_pred: list[float]) -> float:
    return pearson(average_ranks(y_true), average_ranks(y_pred))


def normalized_score(value: float, metadata: dict[str, object]) -> tuple[float, float]:
    baseline = float(metadata["baseline"])
    ideal = float(metadata["ideal"])
    improvement = (value - baseline) / max(1e-12, ideal - baseline)
    return max(0.0, min(1.0, improvement)), improvement


def main() -> int:
    pred_dir = Path(sys.argv[1])
    ref_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {"score": 0.0, "metric_value": 0.0, "improvement": 0.0, "spearman": 0.0}
    try:
        metadata = json.loads((ref_dir / "metadata.json").read_text(encoding="utf-8"))
        if str(metadata.get("evaluation_mode")) != "bio_thermostability_spearman":
            raise ValueError(f"unknown evaluation_mode: {metadata.get('evaluation_mode')!r}")
        answers = read_rows(ref_dir / "answers.csv")
        predictions = read_predictions(pred_dir / "predictions.csv")
        ensure_complete(predictions, answers)
        y_true = [float(row["target"]) for row in answers]
        y_pred = [predictions[row["id"]] for row in answers]
        rho = spearmanr(y_true, y_pred)
        if not math.isfinite(rho):
            rho = 0.0
        score, improvement = normalized_score(rho, metadata)
        scores = {
            "score": score,
            "metric_value": rho,
            "improvement": improvement,
            "spearman": rho,
            "count": float(len(y_true)),
        }
    except Exception as exc:
        (out_dir / "scoring_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    write_json(out_dir / "scores.json", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

