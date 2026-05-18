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
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path.name} is missing a header")
        return list(reader)


def read_predictions(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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
        raise ValueError("predictions.csv has no prediction rows")
    return predictions


def ensure_complete(predictions: dict[str, float], answers: list[dict[str, str]]) -> list[str]:
    expected_ids = [row["id"] for row in answers]
    expected_set = set(expected_ids)
    missing = [row_id for row_id in expected_ids if row_id not in predictions]
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} ids, first: {missing[:3]}")
    extra = [row_id for row_id in predictions if row_id not in expected_set]
    if extra:
        raise ValueError(f"unexpected prediction ids: {extra[:3]}")
    return expected_ids


def causality_effect_estimation(
    predictions: dict[str, float],
    answers: list[dict[str, str]],
    metadata: dict[str, object],
) -> tuple[float, dict[str, float]]:
    ids = ensure_complete(predictions, answers)
    pred_ate = sum(predictions[row_id] for row_id in ids) / len(ids)
    true_effects = [float(row["true_effect"]) for row in answers]
    true_ate = sum(true_effects) / len(true_effects)
    abs_pct_error_of_ate = abs(pred_ate - true_ate) / max(1e-12, abs(true_ate))
    pehe = math.sqrt(
        sum((predictions[row["id"]] - float(row["true_effect"])) ** 2 for row in answers)
        / len(answers)
    )
    details = {
        "abs_pct_error_of_ate": abs_pct_error_of_ate,
        "predicted_ate": pred_ate,
        "true_ate": true_ate,
        "pehe": pehe,
    }
    primary_metric = str(metadata.get("primary_metric", metadata.get("metric", "pehe")))
    if primary_metric == "pehe":
        return pehe, details
    if primary_metric == "abs_pct_error_of_ate":
        return abs_pct_error_of_ate, details
    raise ValueError(f"unknown causality primary_metric: {primary_metric!r}")


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


def main() -> int:
    pred_dir = Path(sys.argv[1])
    ref_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {"score": 0.0, "metric_value": 0.0, "improvement": 0.0}
    try:
        metadata = json.loads((ref_dir / "metadata.json").read_text(encoding="utf-8"))
        mode = str(metadata.get("evaluation_mode", ""))
        if mode != "causality_effect_estimation":
            raise ValueError(f"unknown evaluation_mode: {mode!r}")
        answers = read_rows(ref_dir / "answers.csv")
        predictions = read_predictions(pred_dir / "predictions.csv")
        metric_value, details = causality_effect_estimation(predictions, answers, metadata)
        score, improvement = normalized_score(metric_value, metadata)
        scores = {"score": score, "metric_value": metric_value, "improvement": improvement}
        scores.update(details)
    except Exception as exc:
        (out_dir / "scoring_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    write_json(out_dir / "scores.json", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
