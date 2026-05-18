#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    return parser.parse_args()


def copy_task(task_dir: Path, workspace_task_dir: Path) -> None:
    if workspace_task_dir.exists():
        shutil.rmtree(workspace_task_dir)
    shutil.copytree(task_dir, workspace_task_dir)

    data_dir = workspace_task_dir / "data"
    if data_dir.is_dir():
        for archive_path in sorted(data_dir.glob("*.zip")):
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(workspace_task_dir)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} is missing a header")
        return list(reader.fieldnames), list(reader)


def read_task_config(task_dir: Path) -> dict[str, object]:
    task_json = task_dir / "task.json"
    if not task_json.is_file():
        return {}
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{task_json} must contain a JSON object")
    return payload


def output_columns_from_task(task_config: dict[str, object]) -> list[str]:
    output = task_config.get("output")
    if not isinstance(output, dict):
        return []
    columns = output.get("columns")
    if not isinstance(columns, list):
        return []
    return [str(column) for column in columns]


def validate_ids(rows: list[dict[str, str]], source: Path) -> list[str]:
    ids = [(row.get("id") or "").strip() for row in rows]
    if not ids:
        raise ValueError(f"{source} contains no rows")
    if any(not row_id for row_id in ids):
        raise ValueError(f"{source} contains an empty id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source} contains duplicate ids")
    return ids


def read_test_ids(task_dir: Path, task_config: dict[str, object]) -> list[str]:
    candidates: list[Path] = []
    input_files = task_config.get("input_files")
    if isinstance(input_files, dict):
        for key in ("test", "test_manifest"):
            value = input_files.get(key)
            if isinstance(value, str):
                candidates.append(task_dir / value)
    candidates.append(task_dir / "data" / "test.csv")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            fieldnames, rows = read_csv_rows(candidate)
            if "id" not in fieldnames:
                raise ValueError(f"{candidate} must contain an id column")
            return validate_ids(rows, candidate)
    raise FileNotFoundError(
        "could not find sample_submission.csv or a test manifest with ids"
    )


def is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def infer_numeric_columns(rows: list[dict[str, str]], columns: list[str]) -> set[str]:
    numeric_columns: set[str] = set()
    for column in columns:
        if column == "id":
            continue
        values = [(row.get(column) or "").strip() for row in rows]
        values = [value for value in values if value]
        if values and all(is_finite_number(value) for value in values):
            numeric_columns.add(column)
    return numeric_columns


def read_output_contract(task_dir: Path) -> tuple[list[str], list[str], set[str]]:
    task_config = read_task_config(task_dir)
    sample_path = task_dir / "sample_submission.csv"
    sample_columns: list[str] = []
    sample_rows: list[dict[str, str]] = []

    if sample_path.is_file():
        sample_columns, sample_rows = read_csv_rows(sample_path)

    expected_columns = (
        output_columns_from_task(task_config)
        or sample_columns
        or ["id", "prediction"]
    )
    if "id" not in expected_columns:
        raise ValueError("output columns must include id")
    if sample_columns and sample_columns != expected_columns:
        raise ValueError(
            "sample_submission.csv columns "
            f"{sample_columns} do not match task.json output columns {expected_columns}"
        )

    expected_ids = (
        validate_ids(sample_rows, sample_path)
        if sample_rows
        else read_test_ids(task_dir, task_config)
    )
    numeric_columns = infer_numeric_columns(sample_rows, expected_columns)
    return expected_columns, expected_ids, numeric_columns


def validate_predictions(
    predictions_csv: Path,
    *,
    expected_columns: list[str],
    expected_ids: list[str],
    numeric_columns: set[str],
) -> None:
    if not predictions_csv.is_file():
        raise ValueError(f"{predictions_csv} was not created")

    fieldnames, rows = read_csv_rows(predictions_csv)
    if fieldnames != expected_columns:
        raise ValueError(
            f"predictions.csv columns must be exactly {expected_columns}, got {fieldnames}"
        )

    predictions: dict[str, dict[str, str]] = {}
    for row in rows:
        if None in row:
            raise ValueError("predictions.csv contains rows with too many columns")
        clean_row: dict[str, str] = {}
        for column in expected_columns:
            value = (row.get(column) or "").strip()
            if column == "id":
                if not value:
                    raise ValueError("predictions.csv contains an empty id")
            else:
                if value == "":
                    raise ValueError(
                        f"prediction column {column!r} is empty for id "
                        f"{row.get('id', '').strip()!r}"
                    )
                if column in numeric_columns and not is_finite_number(value):
                    raise ValueError(
                        f"prediction column {column!r} must be finite numeric "
                        f"for id {row.get('id', '').strip()!r}"
                    )
            clean_row[column] = value

        row_id = clean_row["id"]
        if row_id in predictions:
            raise ValueError(f"duplicate prediction id: {row_id}")
        predictions[row_id] = clean_row

    expected = set(expected_ids)
    actual = set(predictions)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"missing {len(missing)} predictions; first missing id: {missing[0]}")
    if extra:
        raise ValueError(f"found {len(extra)} unexpected predictions; first extra id: {extra[0]}")

    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_columns)
        writer.writeheader()
        for row_id in expected_ids:
            writer.writerow(predictions[row_id])


def build_prompt(
    template_path: Path,
    *,
    task_dir: Path,
    workspace_dir: Path,
    workspace_task_dir: Path,
    agent_predictions: Path,
    log_path: Path,
) -> str:
    template = template_path.read_text(encoding="utf-8")
    return template.format(
        task_dir=task_dir,
        workspace_dir=workspace_dir,
        workspace_task_dir=workspace_task_dir,
        agent_predictions=agent_predictions,
        log_path=log_path,
    )


def run_claude(prompt: str, workspace_dir: Path, log_path: Path) -> int:
    claude_bin = os.environ.get("CLAUDE_CLI_BIN", "claude")
    allowed_tools = os.environ.get("CLAUDE_ALLOWED_TOOLS", "Bash,Read,Edit")
    max_turns = os.environ.get("CLAUDE_MAX_TURNS", "30")
    timeout_seconds = int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "3000"))

    cmd = [
        claude_bin,
        "--bare",
        "-p",
        prompt,
        "--permission-mode",
        os.environ.get("CLAUDE_PERMISSION_MODE", "acceptEdits"),
        "--allowedTools",
        allowed_tools,
        "--max-turns",
        max_turns,
        "--output-format",
        os.environ.get("CLAUDE_OUTPUT_FORMAT", "text"),
    ]

    if os.environ.get("CLAUDE_MODEL"):
        cmd.extend(["--model", os.environ["CLAUDE_MODEL"]])
    if os.environ.get("CLAUDE_MAX_BUDGET_USD"):
        cmd.extend(["--max-budget-usd", os.environ["CLAUDE_MAX_BUDGET_USD"]])

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Running command:\n")
        log.write(" ".join(cmd[:3] + ["<prompt>", *cmd[4:]]) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                cmd,
                cwd=workspace_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\nClaude Code timed out after {timeout_seconds} seconds.\n")
            return 124

    return completed.returncode


def main() -> int:
    args = parse_args()
    task_dir = args.task.resolve()
    output_path = args.output.resolve()
    output_dir = args.output_dir.resolve()
    submission_dir = args.submission_dir.resolve()

    workspace_dir = output_dir / "claude_science_ai_workspace"
    workspace_task_dir = workspace_dir / "task"
    agent_predictions = workspace_dir / "predictions.csv"
    log_path = output_dir / "claude_code.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True)

    copy_task(task_dir, workspace_task_dir)
    expected_columns, expected_ids, numeric_columns = read_output_contract(workspace_task_dir)

    prompt = build_prompt(
        submission_dir / "agent_prompt.md",
        task_dir=task_dir,
        workspace_dir=workspace_dir,
        workspace_task_dir=workspace_task_dir,
        agent_predictions=agent_predictions,
        log_path=log_path,
    )
    (workspace_dir / "CLAUDE_TASK.md").write_text(prompt + "\n", encoding="utf-8")

    try:
        claude_returncode = run_claude(prompt, workspace_dir, log_path)
        candidate = agent_predictions
        if not candidate.is_file() and output_path.is_file():
            candidate = output_path
        validate_predictions(
            candidate,
            expected_columns=expected_columns,
            expected_ids=expected_ids,
            numeric_columns=numeric_columns,
        )
        if candidate != output_path:
            shutil.copyfile(candidate, output_path)
        if claude_returncode != 0:
            (output_dir / "claude_nonzero_exit.txt").write_text(
                f"Claude Code exited with status {claude_returncode}, "
                "but a valid predictions.csv was produced.\n"
                f"See {log_path}\n",
                encoding="utf-8",
            )
    except Exception as exc:
        message = textwrap.dedent(
            f"""
            starter submission failed: {exc}

            Check Claude Code logs at:
            {log_path}
            """
        ).strip()
        (output_dir / "baseline_error.txt").write_text(message + "\n", encoding="utf-8")
        print(message, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
