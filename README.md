# ReprodGen

ReprodGen, packaged as `reprodbench`, is a research codebase for generating and evaluating buggy Python code and patched Python code from structured software problem inputs. The runnable entrypoints live in `scripts/common/`, and the reusable implementation lives in `src/reprodbench/`.

## What The Repository Contains

The repository is organized around two experiment families:

1. `buggy_*` pipelines
   These extract semantic guidance such as code intent, functional requirements, and SCOT, then generate buggy code and judge whether it reproduces the target bug.
2. `patched_*` pipelines
   These extract patch guidance, generate patched code, and judge whether the patch fixes the bug.

Most workflows follow the same pattern:

1. Load a CSV dataset.
2. Run one or more LLM-backed extraction or generation stages.
3. Parse structured outputs.
4. Optionally refine after judge or execution feedback.
5. Write results to a CSV in `results/`.

## Repository Layout

- `scripts/common/`
  Main runnable scripts for extraction and reproduction.
- `src/reprodbench/pipeline/`
  Generator, refiner, and judge pipeline classes.
- `src/reprodbench/ablation/`
  Ablation modes and context builders.
- `src/reprodbench/executor/sandbox_client.py`
  HTTP client for the sandbox execution service.
- `src/reprodbench/llm/`
  LLM client and prompt-related code.
- `src/reprodbench/utils/`
  Logging, dataset helpers, and shared utilities.
- `data/`
  Checked-in input datasets.
- `sandbox_executor/`
  FastAPI service used by reproduction scripts to execute generated Python code inside Docker.

## Requirements

- Python `>=3.10,<4.0`
- One LLM backend:
  - Ollama
  - OpenAI
  - Anthropic
- For reproduction scripts: Docker plus the sandbox execution service

Project dependencies are defined in [pyproject.toml](/home/ipd21/Documents/reprodgen-final/pyproject.toml).

## Install

### With `uv`

```bash
uv sync
source .venv/bin/activate
```

For development dependencies:

```bash
uv sync --extra dev
source .venv/bin/activate
```

### With `venv` and `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Environment Variables

Most scripts call `load_dotenv()`, so put a `.env` file in the repo root.

Typical values:

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
SANDBOX_URL=http://localhost:8000
LOG_ENABLED=true
LOG_LEVEL=INFO
LOG_TO_FILE=false
LOG_FILE_PATH=reprodbench.log
```

Example `.env` file:

```env
# Choose the provider keys you actually need
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Sandbox executor used by reproduction scripts
SANDBOX_URL=http://localhost:8000

# Logging
LOG_ENABLED=true
LOG_LEVEL=INFO
LOG_TO_FILE=false
LOG_FILE_PATH=reprodbench.log
```

What they control:

- `OPENAI_API_KEY`
  Required if a script uses `provider="openai"`.
- `ANTHROPIC_API_KEY`
  Required if a script uses `provider="anthropic"`.
- `SANDBOX_URL`
  Required for code reproduction scripts that execute generated code.
- `LOG_ENABLED`, `LOG_LEVEL`, `LOG_TO_FILE`, `LOG_FILE_PATH`
  Used by [src/reprodbench/utils/logger.py](/home/ipd21/Documents/reprodgen-final/src/reprodbench/utils/logger.py).

## Provider Selection

Provider selection is done inside each runner script rather than through a global CLI.

The shared client in `src/reprodbench/llm/client.py` supports:

- `provider="ollama"`
- `provider="openai"`
- `provider="anthropic"`

The checked-in scripts currently default mostly to Ollama-backed model names. If you switch providers, update the generator, refiner, and judge settings consistently inside the same script.

## Sandbox Execution Service

These scripts require a running sandbox service:

- `scripts/common/run_buggy_code_reproduction.py`
- `scripts/common/run_patched_code_reproduction.py`
- `scripts/common/run_patched_code_reproduction_no_answer.py`

They execute generated code through `SandboxExecutorClient`, which calls:

```text
{SANDBOX_URL}/execute
```

The bundled service lives in [sandbox_executor/README.md](/home/ipd21/Documents/reprodgen-final/sandbox_executor/README.md).

### How To Set Up And Use The Sandbox

1. Make sure Docker is installed and the Docker daemon is running.
2. Install the main repo dependencies.
3. In a separate shell, start the sandbox service from `sandbox_executor/`.
4. Set `SANDBOX_URL` in the repo root `.env`.
5. Run one of the reproduction scripts.

Start the sandbox service from `sandbox_executor/` with:

```bash
cd sandbox_executor
uv sync
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Set the repo root `.env` to point at the service:

```env
SANDBOX_URL=http://localhost:8000
```

You can quickly verify that the sandbox is working before running a reproduction script:

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(\"sandbox ok\")",
    "language": "python",
    "python_version": "3.10",
    "requirements": "",
    "timeout_seconds": 30
  }'
```

If the sandbox is working, the response should include `stdout` with `sandbox ok`.

If the sandbox service is not running, extraction scripts can still work, but reproduction scripts that execute code will fail.

## Important Before You Run Anything

Several checked-in scripts still contain hard-coded defaults that do not match the datasets currently present in the repo. In practice, you should expect to edit script-local constants before running an experiment.

Common mismatches include:

- placeholder dataset paths such as `data/run_github_issues_sampled.csv`
- placeholder mappings such as `path` or `path_to_guidance`
- model names that assume a local Ollama setup

This repository is reproducible, but not plug-and-play. Before running a script, confirm that its dataset path, model names, ablation settings, and required input columns match your actual setup.

## Datasets Included In The Repo

- `data/reprodbench-base.csv`
  Starts with columns such as `question_id`, `question`, and `answer`.
- `data/reprodbench-gi.csv`
  Includes columns such as `question_id`, `question_link`, `question_title`, `question_body`, `accepted_answer_body`, `buggy_code`, `patched_code`, `python_version`, `requirements`, and `repository`.
- `data/ReprodBench-verified.xlsx`
  Additional checked-in dataset artifact.

`data/reprodbench-gi.csv` is usually the better starting point for patched reproduction workflows because it already contains runtime and code fields. It is not sufficient by itself for guidance-driven ablation runs, because the guidance columns still need to be added.

## Running The Scripts

There is no single top-level CLI. You run the scripts directly.

General form:

```bash
python scripts/common/<script_name>.py
```

Available main scripts:

```bash
python scripts/common/run_buggy_code_intent_extraction.py
python scripts/common/run_buggy_fr_extraction.py
python scripts/common/run_buggy_scot_extraction.py
python scripts/common/run_buggy_code_reproduction.py
python scripts/common/run_patched_code_intent_extraction.py
python scripts/common/run_patched_functional_requirements_extraction.py
python scripts/common/run_patched_scot_extraction.py
python scripts/common/run_patched_code_reproduction.py
python scripts/common/run_patched_code_reproduction_no_answer.py
```

## Recommended Workflow

### Buggy pipeline

1. Run `run_buggy_code_intent_extraction.py`.
2. Run `run_buggy_fr_extraction.py`.
3. Run `run_buggy_scot_extraction.py`.
4. Merge those outputs into the dataset you want to use for reproduction.
5. Run `run_buggy_code_reproduction.py`.

### Patched pipeline

1. Run `run_patched_code_intent_extraction.py`.
2. Run `run_patched_functional_requirements_extraction.py`.
3. Run `run_patched_scot_extraction.py`.
4. Merge those outputs into the dataset you want to use for patched code generation.
5. Run `run_patched_code_reproduction.py`.

For reproducible comparisons, keep the same prepared CSV, model choice, judge setup, and ablation settings across runs.

## What You Usually Need To Edit

The repo is configured mainly by editing constants near the top of each runner script.

### Generator model

For buggy extraction scripts, edit `GEN_MODELS`.

Example:

```python
GEN_MODELS = [
    "gpt-oss:20b",
]
```

For patched extraction and patched reproduction scripts, edit `MODEL_DATASETS` or `MODEL_DATA_PATHS`.

Example:

```python
MODEL_DATA_PATHS = {
    "gpt-oss:20b": PROJECT_ROOT / "data" / "your_input.csv",
}
```

For reproduction scripts, also update `MODEL_SLUGS` when adding a new model name.

### Judge model

Edit `JUDGE_MODEL_NAME` in the target script.

Example:

```python
JUDGE_MODEL_NAME = "qwen3:8b"
```

### Dataset path

This is the most common edit. Depending on the script, update one of:

- `DATA_PATH`
- `MODEL_DATASETS`
- `MODEL_DATA_PATHS`

### Ablation setup

Reproduction scripts use `ABLATIONS`. Common values are defined in `src/reprodbench/ablation/mode.py`.

Example:

```python
ABLATIONS = [
    AblationMode.NONE,
    AblationMode.CI_FR_SCOT,
]
```

Typical meanings:

- `AblationMode.NONE`
  No semantic guidance.
- `AblationMode.CI`
  Code intent only.
- `AblationMode.FR`
  Functional requirements only.
- `AblationMode.SCOT`
  SCOT only.
- `AblationMode.CI_FR_SCOT`
  Full guidance.

### Output location

Each script defines its own `RESULTS_DIR`. Edit that constant if you want results written elsewhere.

### Retry and timeout knobs

Common script-level settings include:

- `MAX_GENERATOR_RETRIES`
- `MAX_REFINER_RETRIES`
- `MAX_EXEC_RETRIES`
- `MAX_JUDGE_RETRIES`
- `MAX_CYCLES`
- `TIMEOUT_SECONDS`

## Reproduction Input Expectations

The reproduction scripts expect prepared CSVs rather than raw datasets.

For buggy code reproduction, the prepared input usually needs at least:

- `question_id`
- `question_title`
- `question_body`
- `question_link`
- `python_version`
- `requirements`

Guidance columns depend on the chosen ablation mode. For example:

- `AblationMode.CI`
  `buggy_code_intent`
- `AblationMode.FR`
  `buggy_functional_requirements`
- `AblationMode.SCOT`
  `buggy_scot`
- `AblationMode.CI_FR_SCOT`
  `buggy_code_intent`, `buggy_functional_requirements`, `buggy_scot`

Minimal setup for a buggy reproduction run:

1. Run the three buggy extraction scripts.
2. Merge their outputs into one prepared CSV.
3. Point `MODEL_DATA_PATHS[MODEL_NAME]` at that CSV.
4. Make sure the same `MODEL_NAME` exists in `MODEL_SLUGS`.
5. Start the sandbox service.
6. Run `python scripts/common/run_buggy_code_reproduction.py`.

The patched pipeline follows the same pattern, using the patched extraction outputs and patched reproduction script.

## Outputs

Results are written by each script into its configured `RESULTS_DIR`, usually under `results/`. Common output areas include:

- `results/github/buggy_ci/`
- `results/github/buggy_fr/`
- `results/github/buggy_scot/`
- `results/github/buggy_code/`
- `results/github/patched_ci/`
- `results/github/ablation/patched_code/`
- `results/github/ablation/patched_code/no_answer/`
