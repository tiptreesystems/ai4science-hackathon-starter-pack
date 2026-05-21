#!/usr/bin/env python3
"""Run a Codabench-like local evaluation in Docker.

The engine runs participant `run.sh` inside `python:3.11-slim`, with only the
task directory mounted read-only. It then runs the task-owned scorer in a second
container with access to `reference/` and `scoring/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_TRACK = "science_of_ai_ml"
TRACK_ROOTS = {
    "bio": Path("tracks/bio"),
    "materials": Path("tracks/materials"),
    "science_of_ai_ml": Path("tracks/science_of_ai_ml"),
}
DEFAULT_PASS_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_MODEL",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
    "CLAUDE_MAX_TURNS",
    "CLAUDE_TIMEOUT_SECONDS",
    "CLAUDE_ALLOWED_TOOLS",
    "CLAUDE_PERMISSION_MODE",
    "CLAUDE_OUTPUT_FORMAT",
    "CLAUDE_MAX_BUDGET_USD",
    "BASELINE_SKIP_PIP_INSTALL",
    "BASELINE_INSTALL_CLAUDE",
    "BASELINE_INSTALL_NODE",
)
SECRET_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "CLAUDE_API_KEY",
    "LITELLM_API_KEY",
    "OPENAI_API_KEY",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("starter-submission"),
        help="Submission directory or zip with run.sh at its root.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help=(
            "Task packet to run, for example train_01 or validation_01. A path "
            "to a packet root is also accepted. May be repeated."
        ),
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Run every train_* packet under --track-root.",
    )
    parser.add_argument(
        "--track",
        choices=sorted(TRACK_ROOTS),
        default=DEFAULT_TRACK,
        help="Track to run when --task is a packet name.",
    )
    parser.add_argument(
        "--track-root",
        type=Path,
        default=None,
        help="Directory containing task packets. Overrides --track.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for extracted submissions, outputs, logs, and scores.",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image to run.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-container timeout in seconds.")
    parser.add_argument(
        "--network",
        default="bridge",
        help="Docker network mode. Use none to check offline behavior.",
    )
    parser.add_argument("--pull", action="store_true", help="Pull --image before running.")
    parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        help="Additional host environment variable to pass into the submission container.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Explicit environment variable to pass into the submission container.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (root / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe path in zip: {info.filename}")
        zf.extractall(root)


def prepare_submission(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    elif source.is_file():
        safe_extract_zip(source, destination)
    else:
        raise FileNotFoundError(source)

    run_sh = destination / "run.sh"
    if not run_sh.is_file():
        raise FileNotFoundError(f"submission missing run.sh at root: {run_sh}")
    run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def resolve_tasks(track_root: Path, task_args: list[str], all_train: bool) -> list[Path]:
    if all_train:
        task_args.extend(path.name for path in sorted(track_root.glob("train_*")) if path.is_dir())
    if not task_args:
        raise ValueError("provide --task train_01, --task validation_01, or --all-train")

    tasks: list[Path] = []
    seen: set[Path] = set()
    for raw in task_args:
        candidate = Path(raw)
        if not candidate.exists():
            candidate = track_root / raw
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        for required in ("task/task.json", "reference/answers.csv", "scoring/score.py"):
            if not (candidate / required).is_file():
                raise FileNotFoundError(f"{candidate} is missing {required}")
        tasks.append(candidate)
    return tasks


def task_output_files(task_packet: Path) -> list[str]:
    task_json = task_packet / "task" / "task.json"
    if not task_json.is_file():
        return ["predictions.csv"]
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    output = payload.get("output") if isinstance(payload, dict) else None
    if not isinstance(output, dict):
        return ["predictions.csv"]
    files = output.get("files")
    if isinstance(files, list) and files and all(isinstance(item, str) for item in files):
        return [str(item) for item in files]
    file_name = output.get("file")
    if isinstance(file_name, str) and file_name:
        return [file_name]
    return ["predictions.csv"]


def docker_env_args(explicit: list[str], pass_env: list[str]) -> list[str]:
    args: list[str] = []
    for name in [*DEFAULT_PASS_ENV, *pass_env]:
        if name in os.environ:
            args.extend(["--env", name])
    for item in explicit:
        if "=" not in item:
            raise ValueError(f"--env must be NAME=VALUE, got {item!r}")
        args.extend(["--env", item])
    return args


def redacted_cmd(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for token in cmd:
        if redact_next:
            name = token.split("=", 1)[0]
            redacted.append(f"{name}=<redacted>" if name in SECRET_ENV_NAMES else token)
            redact_next = False
            continue
        redacted.append(token)
        if token in {"--env", "-e"}:
            redact_next = True
    return redacted


def run_logged(cmd: list[str], log_path: Path, timeout: int, container_name: str) -> int:
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("Running command:\n")
            log.write(" ".join(redacted_cmd(cmd)) + "\n\n")
            log.flush()
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", container_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nTimed out after {timeout} seconds.\n")
        return 124
    finally:
        elapsed = time.time() - start
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nElapsed seconds: {elapsed:.3f}\n")
    return result.returncode


def run_submission_container(
    *,
    image: str,
    network: str,
    timeout: int,
    submission_dir: Path,
    task_packet: Path,
    output_dir: Path,
    log_path: Path,
    env_args: list[str],
) -> int:
    name = f"ai4science-submission-{task_packet.parent.name}-{task_packet.name}-{os.getpid()}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--cap-drop",
        "CHOWN",
        "--network",
        network,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp/ai4science-home",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONUNBUFFERED=1",
        *env_args,
        "--volume",
        f"{submission_dir}:/app/ingested_program:rw",
        "--volume",
        f"{task_packet / 'task'}:/app/input_data:ro",
        "--volume",
        f"{output_dir}:/app/output:rw",
        "--workdir",
        "/app/ingested_program",
        image,
        "bash",
        "-lc",
        (
            "chmod +x ./run.sh && "
            "./run.sh --task /app/input_data "
            "--output /app/output/predictions.csv "
            "--output-dir /app/output"
        ),
    ]
    return run_logged(cmd, log_path, timeout, name)


def run_scoring_container(
    *,
    image: str,
    network: str,
    timeout: int,
    task_packet: Path,
    prediction_dir: Path,
    score_dir: Path,
    log_path: Path,
    pip_cache_dir: Path,
) -> int:
    name = f"ai4science-scoring-{task_packet.parent.name}-{task_packet.name}-{os.getpid()}"
    pip_cache_dir.mkdir(parents=True, exist_ok=True)
    scoring_command = (
        "set -euo pipefail; "
        "if [ -f /app/program/requirements.txt ]; then "
        "python -m pip install --quiet --disable-pip-version-check "
        "--target /tmp/ai4science-scoring-site -r /app/program/requirements.txt; "
        "export PYTHONPATH=/tmp/ai4science-scoring-site${PYTHONPATH:+:$PYTHONPATH}; "
        "fi; "
        "python /app/program/score.py /app/input/res /app/input/ref /app/output"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--cap-drop",
        "CHOWN",
        "--network",
        network,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp/ai4science-home",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "PIP_CACHE_DIR=/tmp/pip-cache",
        "--volume",
        f"{prediction_dir}:/app/input/res:ro",
        "--volume",
        f"{task_packet / 'reference'}:/app/input/ref:ro",
        "--volume",
        f"{task_packet / 'scoring'}:/app/program:ro",
        "--volume",
        f"{score_dir}:/app/output:rw",
        "--volume",
        f"{pip_cache_dir}:/tmp/pip-cache:rw",
        image,
        "bash",
        "-lc",
        scoring_command,
    ]
    return run_logged(cmd, log_path, timeout, name)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def run_task(
    *,
    task_packet: Path,
    submission_source: Path,
    work_dir: Path,
    image: str,
    network: str,
    timeout: int,
    env_args: list[str],
) -> dict[str, object]:
    track_name = task_packet.parent.name
    task_name = task_packet.name
    task_work = work_dir / track_name / task_name
    submission_dir = task_work / "submission"
    output_dir = task_work / "submission_output"
    score_dir = task_work / "score_output"
    logs_dir = task_work / "logs"
    for path in (output_dir, score_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    prepare_submission(submission_source, submission_dir)

    submission_status = run_submission_container(
        image=image,
        network=network,
        timeout=timeout,
        submission_dir=submission_dir,
        task_packet=task_packet,
        output_dir=output_dir,
        log_path=logs_dir / "submission.log",
        env_args=env_args,
    )

    scoring_status: int | None = None
    scores: object | None = None
    expected_outputs = task_output_files(task_packet)
    outputs_present = all((output_dir / rel_path).is_file() for rel_path in expected_outputs)
    if submission_status == 0 and outputs_present:
        scoring_status = run_scoring_container(
            image=image,
            network=network,
            timeout=timeout,
            task_packet=task_packet,
            prediction_dir=output_dir,
            score_dir=score_dir,
            log_path=logs_dir / "scoring.log",
            pip_cache_dir=work_dir / "_scoring_pip_cache",
        )
        scores_path = score_dir / "scores.json"
        if scores_path.is_file():
            scores = read_json(scores_path)

    missing_outputs = [
        rel_path for rel_path in expected_outputs if not (output_dir / rel_path).is_file()
    ]
    scoring_error_path = score_dir / "scoring_error.txt"
    scoring_error = (
        scoring_error_path.read_text(encoding="utf-8").strip()
        if scoring_error_path.is_file()
        else None
    )
    success = submission_status == 0 and not missing_outputs and scoring_status == 0 and scoring_error is None
    result = {
        "task": f"{track_name}/{task_name}",
        "expected_outputs": expected_outputs,
        "missing_outputs": missing_outputs,
        "success": success,
        "submission_status": submission_status,
        "scoring_status": scoring_status,
        "scoring_error": scoring_error,
        "scores": scores,
        "work_dir": str(task_work),
        "submission_log": str(logs_dir / "submission.log"),
        "scoring_log": str(logs_dir / "scoring.log"),
    }
    (task_work / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    root = repo_root()
    submission = resolve_path(args.submission, root)
    track_root_arg = args.track_root or TRACK_ROOTS[args.track]
    track_root = resolve_path(track_root_arg, root)
    tasks = resolve_tasks(track_root, list(args.task), args.all_train)
    work_dir = (
        resolve_path(args.work_dir, Path.cwd())
        if args.work_dir
        else root / "eval-runs" / timestamp()
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.pull:
        subprocess.run(["docker", "pull", args.image], check=True)

    env_args = docker_env_args(args.env, args.pass_env)
    results = [
        run_task(
            task_packet=task,
            submission_source=submission,
            work_dir=work_dir,
            image=args.image,
            network=args.network,
            timeout=args.timeout,
            env_args=env_args,
        )
        for task in tasks
    ]

    summary = {"image": args.image, "work_dir": str(work_dir), "results": results}
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    failed = [item for item in results if not item["success"]]
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
