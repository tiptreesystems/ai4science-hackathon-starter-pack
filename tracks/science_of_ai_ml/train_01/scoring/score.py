#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def write_scores(out_dir: Path, payload: dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_answers(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "target"]:
            raise ValueError("answers.csv must have columns id,target")

        answers: dict[str, int] = {}
        for row in reader:
            row_id = row["id"].strip()
            if not row_id:
                raise ValueError("answers.csv contains an empty id")
            if row_id in answers:
                raise ValueError(f"duplicate answer id: {row_id}")
            target = int(row["target"])
            if target not in (0, 1):
                raise ValueError(f"answer for {row_id} is not binary")
            answers[row_id] = target

    if not answers:
        raise ValueError("answers.csv contains no rows")
    return answers


def parse_prediction(value: str, row_id: str) -> int:
    text = value.strip()
    if text in {"0", "1"}:
        return int(text)

    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"prediction for {row_id} is not numeric") from exc

    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"prediction for {row_id} must be 0 or 1")
    return int(numeric)


def read_predictions(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise ValueError("predictions.csv was not found")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "prediction"]:
            raise ValueError("predictions.csv must contain exactly id,prediction columns")

        predictions: dict[str, int] = {}
        for row in reader:
            row_id = row["id"].strip()
            if not row_id:
                raise ValueError("predictions.csv contains an empty id")
            if row_id in predictions:
                raise ValueError(f"duplicate prediction id: {row_id}")
            predictions[row_id] = parse_prediction(row["prediction"], row_id)

    if not predictions:
        raise ValueError("predictions.csv contains no rows")
    return predictions


def score(predictions: dict[str, int], answers: dict[str, int], metadata: dict[str, object]) -> dict[str, float]:
    expected_ids = set(answers)
    predicted_ids = set(predictions)

    missing = sorted(expected_ids - predicted_ids)
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} ids; first missing id: {missing[0]}")

    unexpected = sorted(predicted_ids - expected_ids)
    if unexpected:
        raise ValueError(f"predictions.csv contains {len(unexpected)} unexpected ids; first unexpected id: {unexpected[0]}")

    correct = sum(predictions[row_id] == target for row_id, target in answers.items())
    accuracy = correct / len(answers)

    baseline = float(metadata.get("baseline", 0.5))
    ideal = float(metadata.get("ideal", 1.0))
    improvement = (accuracy - baseline) / max(1e-12, ideal - baseline)
    normalized = max(0.0, min(1.0, improvement))
    return {
        "score": normalized,
        "metric_value": accuracy,
        "heldout_accuracy": accuracy,
        "improvement": improvement,
        "correct": float(correct),
        "total": float(len(answers)),
    }


def main() -> int:
    pred_dir = Path(sys.argv[1])
    ref_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])

    scores = {
        "score": 0.0,
        "metric_value": 0.0,
        "heldout_accuracy": 0.0,
        "improvement": 0.0,
        "correct": 0.0,
        "total": 0.0,
    }
    try:
        metadata = json.loads((ref_dir / "metadata.json").read_text(encoding="utf-8"))
        answers = read_answers(ref_dir / "answers.csv")
        predictions = read_predictions(pred_dir / "predictions.csv")
        scores = score(predictions, answers, metadata)
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scoring_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    write_scores(out_dir, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
