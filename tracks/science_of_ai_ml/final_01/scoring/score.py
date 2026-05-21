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


def binary_label(value: float) -> int:
    return 1 if value >= 0.5 else 0


def ensure_binary_predictions(predictions: dict[str, float]) -> None:
    for row_id, value in predictions.items():
        if value not in (0.0, 1.0):
            raise ValueError(f"prediction for {row_id} must be 0 or 1")


def integer_label(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("prediction is not finite")
    return int(round(value))


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


def classification_accuracy(predictions: dict[str, float], answers: list[dict[str, str]]) -> tuple[float, dict[str, float]]:
    ids = ensure_complete(predictions, answers)
    correct = 0
    for row in answers:
        if integer_label(predictions[row["id"]]) == int(row["target"]):
            correct += 1
    accuracy = correct / len(ids)
    return accuracy, {"accuracy": accuracy, "coverage": len(ids) / len(answers)}


def continual_average_accuracy(predictions: dict[str, float], answers: list[dict[str, str]]) -> tuple[float, dict[str, float]]:
    ensure_complete(predictions, answers)
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in answers:
        by_task[row["task_id"]].append(row)

    task_acc: dict[str, float] = {}
    for task_id, rows in sorted(by_task.items()):
        correct = sum(integer_label(predictions[row["id"]]) == int(row["target"]) for row in rows)
        task_acc[task_id] = correct / len(rows)

    average_acc = sum(task_acc.values()) / len(task_acc)
    details = {"average_acc": average_acc}
    details.update({f"acc_{task_id}": value for task_id, value in task_acc.items()})
    return average_acc, details


def fairness_abs_aod(predictions: dict[str, float], answers: list[dict[str, str]], metadata: dict[str, object]) -> tuple[float, dict[str, float]]:
    ensure_complete(predictions, answers)
    ensure_binary_predictions(predictions)
    privileged = str(metadata.get("privileged_group", "privileged"))
    unprivileged = str(metadata.get("unprivileged_group", "unprivileged"))

    def rates(group: str) -> tuple[float, float]:
        rows = [row for row in answers if row["protected_group"] == group]
        positives = [row for row in rows if int(row["target"]) == 1]
        negatives = [row for row in rows if int(row["target"]) == 0]
        if not positives or not negatives:
            raise ValueError(f"group {group!r} needs positive and negative reference rows")
        tp = sum(binary_label(predictions[row["id"]]) == 1 for row in positives)
        fp = sum(binary_label(predictions[row["id"]]) == 1 for row in negatives)
        return tp / len(positives), fp / len(negatives)

    tpr_unpriv, fpr_unpriv = rates(unprivileged)
    tpr_priv, fpr_priv = rates(privileged)
    aod = 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))
    correct = sum(binary_label(predictions[row["id"]]) == int(row["target"]) for row in answers)
    accuracy = correct / len(answers)
    return abs(aod), {
        "abs_aod": abs(aod),
        "aod": aod,
        "accuracy": accuracy,
        "tpr_unprivileged": tpr_unpriv,
        "fpr_unprivileged": fpr_unpriv,
        "tpr_privileged": tpr_priv,
        "fpr_privileged": fpr_priv,
    }


def causality_effect_estimation(
    predictions: dict[str, float],
    answers: list[dict[str, str]],
    metadata: dict[str, object],
) -> tuple[float, dict[str, float]]:
    ids = ensure_complete(predictions, answers)
    pred_ate = sum(predictions[row_id] for row_id in ids) / len(ids)
    true_effects = [float(row["true_effect"]) for row in answers]
    true_ate = sum(true_effects) / len(true_effects)
    error = abs(pred_ate - true_ate) / max(1e-12, abs(true_ate))
    pehe = math.sqrt(
        sum((predictions[row["id"]] - float(row["true_effect"])) ** 2 for row in answers)
        / len(answers)
    )
    details = {
        "abs_pct_error_of_ate": error,
        "predicted_ate": pred_ate,
        "true_ate": true_ate,
        "pehe": pehe,
    }
    primary_metric = str(metadata.get("primary_metric", metadata.get("metric", "abs_pct_error_of_ate")))
    if primary_metric == "abs_pct_error_of_ate":
        return error, details
    if primary_metric == "pehe":
        return pehe, details
    raise ValueError(f"unknown causality primary_metric: {primary_metric!r}")


def causality_abs_pct_ate_error(predictions: dict[str, float], answers: list[dict[str, str]]) -> tuple[float, dict[str, float]]:
    return causality_effect_estimation(predictions, answers, {"primary_metric": "abs_pct_error_of_ate"})


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


def apply_utility_gate(score: float, details: dict[str, float], metadata: dict[str, object]) -> float:
    if metadata.get("score_utility") != "accuracy_gate":
        return score
    accuracy = details.get("accuracy")
    if accuracy is None:
        raise ValueError("accuracy_gate requires the scorer to compute accuracy")
    floor = float(metadata.get("utility_floor", 0.0))
    reference = float(metadata.get("baseline_accuracy", 1.0))
    utility = (accuracy - floor) / max(1e-12, reference - floor)
    return score * max(0.0, min(1.0, utility))


def compute_metric(predictions: dict[str, float], answers: list[dict[str, str]], metadata: dict[str, object]) -> tuple[float, dict[str, float]]:
    mode = str(metadata["evaluation_mode"])
    if mode == "classification_accuracy":
        return classification_accuracy(predictions, answers)
    if mode == "continual_average_accuracy":
        return continual_average_accuracy(predictions, answers)
    if mode == "fairness_abs_aod":
        return fairness_abs_aod(predictions, answers, metadata)
    if mode == "causality_effect_estimation":
        return causality_effect_estimation(predictions, answers, metadata)
    if mode == "causality_abs_pct_ate_error":
        return causality_abs_pct_ate_error(predictions, answers)
    raise ValueError(f"unknown evaluation_mode: {mode!r}")


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
        metric_value, details = compute_metric(predictions, answers, metadata)
        score, improvement = normalized_score(metric_value, metadata)
        score = apply_utility_gate(score, details, metadata)
        scores = {"score": score, "metric_value": metric_value, "improvement": improvement}
        scores.update(details)
    except Exception as exc:
        (out_dir / "scoring_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    write_json(out_dir / "scores.json", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
