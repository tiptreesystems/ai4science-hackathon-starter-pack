# AI4Science Research Agent Starter Submission

This directory is a starter Codabench submission. The submitted zip must contain
`run.sh` directly at the zip root.

Do not zip the enclosing `starter-submission/` folder itself. A zip whose
contents start with `starter-submission/run.sh` will fail because Codabench only
looks for `run.sh` at the archive root.

```bash
# From the starter-pack repository root:
cd starter-submission
zip -r ../ai4science-research-agent-starter-submission.zip .
```

After packaging, this should be true:

```text
run.sh
README.md
requirements.txt
agent_prompt.md
scripts/run_claude_agent.py
```

Codabench invokes:

```bash
./run.sh --task /path/to/task --output /path/to/predictions.csv --output-dir /path/to/output
```

The runner installs the small Python dependency set in `requirements.txt`,
stages the task in a writable workspace, extracts zip archives under `data/`,
and runs Claude Code in headless mode with `claude --bare -p ...`. The research
agent reads the task's `task.md` and `task.json`, solves the research task
described there, and writes `predictions.csv`. The wrapper validates the
declared output columns and test IDs before copying the file to the requested
output path.

Expected environment:

- `ANTHROPIC_API_KEY` available to Claude Code.
- Optional `ANTHROPIC_BASE_URL` if using an Anthropic-compatible gateway.
- Optional `.env.secrets` in this directory. `run.sh` sources it before starting
  the agent; use it for local API settings and do not put real keys in Git.
- `claude` on `PATH`, or network access so `run.sh` can install Claude Code.
  If npm is unavailable, `run.sh` downloads Node.js into a runtime directory
  under `/tmp` and uses the bundled npm.
- Network access if Python dependencies, Node.js, or Claude Code need to be
  installed at runtime.

Useful overrides:

- `BASELINE_SKIP_PIP_INSTALL=1`: skip Python dependency installation.
- `BASELINE_INSTALL_CLAUDE=0`: fail instead of installing Claude Code when
  `claude` is missing.
- `BASELINE_INSTALL_NODE=0`: fail instead of downloading Node.js when npm is
  missing.
- `BASELINE_RUNTIME_DIR=/path/to/runtime`: store runtime-only installs and
  caches somewhere other than the default `/tmp/science-ai-baseline-runtime`.
- `BASELINE_NODE_MAJOR=22`: Node.js major version to download when npm is
  missing.
- `BASELINE_NODE_DIR=/path/to/node`: reuse or install Node.js in a specific
  directory.
- `CLAUDE_CLI_BIN=/path/to/claude`: use a specific Claude Code binary.
- `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=0`: re-enable Claude Code beta
  headers if your endpoint supports them.
- `CLAUDE_MODEL=...`: request a specific Claude model.
- `CLAUDE_MAX_TURNS=30`: cap agent turns.
- `CLAUDE_TIMEOUT_SECONDS=3000`: cap the headless agent call.
- `CLAUDE_MAX_BUDGET_USD=...`: pass a Claude Code budget limit.

This baseline does not include hidden labels or scorer files. It only consumes
the participant-visible task directory passed by Codabench.
