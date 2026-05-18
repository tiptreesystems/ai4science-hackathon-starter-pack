# AI4Science Hackathon Starter Pack

This repository contains participant-facing starter materials.

## Contents

- `starter-submission/`: a ready-to-zip Codabench research-agent submission.
- `tracks/science_of_ai_ml/`: Science of AI / ML practice and validation task
  packets, including visible task files, references, and scoring programs.
- `scripts/eval_engine.py`: a Docker-based local evaluator that mirrors the
  Codabench `run.sh` contract on the `python:3.11-slim` worker image.

To create a Codabench submission zip:

```bash
cd starter-submission
zip -r ../ai4science-research-agent-starter-submission.zip .
```

The zip root must contain `run.sh`.

## Available Tasks

The starter pack includes these Science of AI / ML packets:

- `train_01`: Colored digit domain generalization.
- `train_02`: Backdoor-robust digit classification.
- `validation_01`: Synthetic treatment-effect estimation.

Each packet contains:

```text
task/        # Files visible to the submitted run.sh
reference/   # Local answer data for this starter pack
scoring/     # Local scoring program
```

No `final_*` packets are included.

## Local Eval Engine

Use the Docker eval engine before uploading. It runs the submission inside
`python:3.11-slim`, with the task directory mounted read-only, then runs the
task scorer against the generated `predictions.csv`.

Requirements:

- Docker installed and running.
- Network access if your submission installs dependencies or calls an API.

Run one train task:

```bash
python3 scripts/eval_engine.py --task train_01
```

Run all Science of AI / ML train tasks:

```bash
python3 scripts/eval_engine.py --all-train
```

Run the validation task:

```bash
python3 scripts/eval_engine.py --task validation_01
```

Run a packaged submission zip:

```bash
python3 scripts/eval_engine.py \
  --submission ai4science-research-agent-starter-submission.zip \
  --task train_01
```

Run without network to check whether your submission is self-contained:

```bash
python3 scripts/eval_engine.py --task train_01 --network none
```

The default starter submission reads secrets from
`starter-submission/.env.secrets`. The eval engine also passes common Claude
and LiteLLM environment variables from your shell into the submission container,
including `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and `CLAUDE_MODEL`.

Outputs, logs, and scores are written under `eval-runs/`, with one subdirectory
per task. Check `logs/submission.log`, `logs/scoring.log`, and `result.json`
when debugging.

If a submission works there, it is using the same `run.sh` contract and base
worker image expected by Codabench. The local engine is still a simulator:
server-side time limits, queue settings, and hidden final tasks are enforced by
Codabench.
