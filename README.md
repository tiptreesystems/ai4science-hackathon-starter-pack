# AI4Science Hackathon Starter Pack

This repository contains participant-facing starter materials.

## Contents

- `starter-submission/`: a ready-to-zip Codabench research-agent submission.
- `tracks/`: released task packets for Science of AI / ML, Materials, and
  Bio, including visible task files plus local references and scoring programs
  for starter-pack testing.
- `scripts/eval_engine.py`: a Docker-based local evaluator that mirrors the
  Codabench `run.sh` contract on the `python:3.11-slim` worker image.

To create a Codabench submission zip:

```bash
cd starter-submission
zip -r ../ai4science-research-agent-starter-submission.zip .
```

The zip root must contain `run.sh`.

## Available Tasks

The starter pack includes these released packets:

Science of AI / ML:

- `train_01`: Colored digit domain generalization.
- `train_02`: Backdoor-robust digit classification.
- `validation_01`: Synthetic treatment-effect estimation.
- `final_01`: Fair tabular binary classification.

Bio:

- `train_01`: Enzyme-substrate activity prediction.
- `validation_01`: Peptide-MHC class I binding affinity prediction.
- `final_01`: Protein thermostability mutation prediction.

Materials:

- `train_01`: Descriptor engineering for methane-conversion catalysts.
- `validation_01`: Descriptor engineering for solid electrolytes.
- `final_01`: Descriptor engineering for CO2RR electrocatalysts.

Each packet contains:

```text
task/        # Files visible to the submitted run.sh
reference/   # Local answer data for this starter pack
scoring/     # Local scoring program
```

All released hackathon packets are included.

## Local Eval Engine

Use the Docker eval engine before uploading. It runs the submission inside
`python:3.11-slim`, with the task directory mounted read-only, then runs the
task scorer against the generated output file or files declared by `task.json`.
If a scorer declares `scoring/requirements.txt`, the engine installs those
packages into the scoring container before running `score.py`.

Requirements:

- Docker installed and running.
- Network access if your submission installs dependencies or calls an API.

Run one train task:

```bash
python3 scripts/eval_engine.py --task train_01
```

Run a specific track:

```bash
python3 scripts/eval_engine.py --track bio --task validation_01
python3 scripts/eval_engine.py --track materials --task train_01
```

Run all train tasks in a track:

```bash
python3 scripts/eval_engine.py --all-train
python3 scripts/eval_engine.py --track bio --all-train
```

Run the validation task:

```bash
python3 scripts/eval_engine.py --task validation_01
```

Run a final task:

```bash
python3 scripts/eval_engine.py --track bio --task final_01
```

Run a packaged submission zip:

```bash
python3 scripts/eval_engine.py \
  --submission ai4science-research-agent-starter-submission.zip \
  --task train_01
```

Run without network to check whether your submission is self-contained. Tasks
whose scorers declare extra Python requirements may still need the default
networked mode so the local scoring container can install scorer dependencies.

```bash
python3 scripts/eval_engine.py --task train_01 --network none
```

The default starter submission reads secrets from
`starter-submission/.env.secrets`. The eval engine also passes common Claude
and LiteLLM environment variables from your shell into the submission container,
including `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and `CLAUDE_MODEL`.

Outputs, logs, and scores are written under `eval-runs/`, grouped by
`track/task`. Check `logs/submission.log`, `logs/scoring.log`, and
`result.json` when debugging. If the scorer writes `scoring_error.txt`, the
local engine marks that task as unsuccessful and includes the error text in
`result.json`.

If a submission works there, it is using the same `run.sh` contract and base
worker image expected by Codabench. The local engine is still a simulator:
server-side time limits and queue settings are enforced by Codabench.
