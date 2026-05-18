#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
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
            if value < 0.0 or value > 1.0:
                raise ValueError(f"prediction for {row_id} must be in [0, 1]")
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


def average_precision_score(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0

    grouped: dict[float, list[int]] = defaultdict(list)
    for label, score in zip(labels, scores):
        grouped[score].append(label)

    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positive += sum(group)
        false_positive += len(group) - sum(group)
        recall = true_positive / positives
        precision = true_positive / max(1, true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


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

    scores: dict[str, float] = {"score": 0.0, "metric_value": 0.0, "improvement": 0.0, "auprc": 0.0}
    try:
        metadata = json.loads((ref_dir / "metadata.json").read_text(encoding="utf-8"))
        if str(metadata.get("evaluation_mode")) != "bio_enzyme_substrate_auprc":
            raise ValueError(f"unknown evaluation_mode: {metadata.get('evaluation_mode')!r}")
        answers = read_rows(ref_dir / "answers.csv")
        predictions = read_predictions(pred_dir / "predictions.csv")
        ensure_complete(predictions, answers)
        labels = [int(row["target"]) for row in answers]
        pred_scores = [predictions[row["id"]] for row in answers]
        auprc = average_precision_score(labels, pred_scores)
        score, improvement = normalized_score(auprc, metadata)
        scores = {
            "score": score,
            "metric_value": auprc,
            "improvement": improvement,
            "auprc": auprc,
            "positive_rate": sum(labels) / len(labels),
            "count": float(len(labels)),
        }
    except Exception as exc:
        (out_dir / "scoring_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    write_json(out_dir / "scores.json", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

