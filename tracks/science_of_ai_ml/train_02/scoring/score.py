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
            predictions[row_id] = float(row["prediction"])
    if not predictions:
        raise ValueError("predictions.csv contains no rows")
    return predictions


def integer_label(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("prediction is not finite")
    rounded = int(round(value))
    if abs(value - rounded) > 1e-9 or rounded < 0 or rounded > 9:
        raise ValueError("predictions must be integer digit labels from 0 through 9")
    return rounded


def ensure_complete(predictions: dict[str, float], answers: list[dict[str, str]]) -> None:
    expected_ids = [row["id"] for row in answers]
    expected_set = set(expected_ids)
    missing = [row_id for row_id in expected_ids if row_id not in predictions]
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} ids, first: {missing[:3]}")
    extra = [row_id for row_id in predictions if row_id not in expected_set]
    if extra:
        raise ValueError(f"unexpected prediction ids: {extra[:3]}")


def accuracy(rows: list[dict[str, str]], predictions: dict[str, float]) -> float:
    if not rows:
        raise ValueError("cannot compute accuracy on an empty slice")
    correct = sum(integer_label(predictions[row["id"]]) == int(row["target"]) for row in rows)
    return correct / len(rows)


def normalized_score(value: float, metadata: dict[str, object]) -> tuple[float, float]:
    baseline = float(metadata["baseline"])
    ideal = float(metadata["ideal"])
    direction = str(metadata["direction"])
    if direction == "higher":
        improvement = (value - baseline) / max(1e-12, ideal - baseline)
    elif direction == "lower":
        improvement = (baseline - value) / max(1e-12, baseline - ideal)
    else:
        raise ValueError(f"unknown metric direction: {direction!r}")
    return max(0.0, min(1.0, improvement)), improvement


def compute_scores(
    predictions: dict[str, float],
    answers: list[dict[str, str]],
    metadata: dict[str, object],
) -> dict[str, float]:
    ensure_complete(predictions, answers)
    clean_rows = [row for row in answers if row.get("slice") == "clean"]
    seen_trigger_rows = [row for row in answers if row.get("slice") == "seen_trigger"]
    unseen_trigger_rows = [row for row in answers if row.get("slice") == "unseen_trigger"]
    if not clean_rows or not seen_trigger_rows or not unseen_trigger_rows:
        raise ValueError("answers.csv must contain clean, seen_trigger, and unseen_trigger slices")

    clean_acc = accuracy(clean_rows, predictions)
    seen_trigger_acc = accuracy(seen_trigger_rows, predictions)
    unseen_trigger_acc = accuracy(unseen_trigger_rows, predictions)
    clean_weight = float(metadata.get("clean_weight", 0.5))
    seen_weight = float(metadata.get("seen_trigger_weight", 0.3))
    unseen_weight = float(metadata.get("unseen_trigger_weight", 0.2))
    defense_score = (
        clean_weight * clean_acc
        + seen_weight * seen_trigger_acc
        + unseen_weight * unseen_trigger_acc
    )

    stress_rows = seen_trigger_rows + unseen_trigger_rows
    attack_success_rate = (
        sum(integer_label(predictions[row["id"]]) == int(row["attack_label"]) for row in stress_rows)
        / len(stress_rows)
    )
    seen_attack_success_rate = (
        sum(integer_label(predictions[row["id"]]) == int(row["attack_label"]) for row in seen_trigger_rows)
        / len(seen_trigger_rows)
    )
    unseen_attack_success_rate = (
        sum(integer_label(predictions[row["id"]]) == int(row["attack_label"]) for row in unseen_trigger_rows)
        / len(unseen_trigger_rows)
    )

    score, improvement = normalized_score(defense_score, metadata)
    return {
        "score": score,
        "metric_value": defense_score,
        "improvement": improvement,
        "defense_score": defense_score,
        "clean_acc": clean_acc,
        "seen_trigger_acc": seen_trigger_acc,
        "unseen_trigger_acc": unseen_trigger_acc,
        "attack_success_rate": attack_success_rate,
        "seen_attack_success_rate": seen_attack_success_rate,
        "unseen_attack_success_rate": unseen_attack_success_rate,
        "clean_count": float(len(clean_rows)),
        "seen_trigger_count": float(len(seen_trigger_rows)),
        "unseen_trigger_count": float(len(unseen_trigger_rows)),
    }


def main() -> int:
    pred_dir = Path(sys.argv[1])
    ref_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {"score": 0.0, "metric_value": 0.0, "improvement": 0.0}
    try:
        metadata = json.loads((ref_dir / "metadata.json").read_text(encoding="utf-8"))
        answers = read_rows(ref_dir / "answers.csv")
        predictions = read_predictions(pred_dir / "predictions.csv")
        if str(metadata.get("evaluation_mode")) != "backdoor_robust_digit_classification":
            raise ValueError(f"unknown evaluation_mode: {metadata.get('evaluation_mode')!r}")
        scores = compute_scores(predictions, answers, metadata)
    except Exception as exc:
        (out_dir / "scoring_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    write_json(out_dir / "scores.json", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
