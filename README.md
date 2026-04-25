# ReprodGen

ReprodBench is a repository for generating and evaluating buggy Python code and patched Python code from structured software problem inputs.

The code you actually run lives mostly in `scripts/common/`. The reusable implementation lives in `src/reprodbench/`.

## What The Project Does

The repository has two main experiment families:

1. `buggy_*` pipelines
   These extract guidance such as code intent, functional requirements, and SCOT, then generate buggy code and judge whether it reproduces the target bug.
2. `patched_*` pipelines
   These extract patch guidance, generate patched code, and judge whether the patch fixes the bug.

Most runs follow this pattern:

1. Load a CSV dataset from `data/` or another file you point the script to.
2. Call an LLM through LangChain.
3. Parse structured output.
4. Optionally refine after judge feedback or execution feedback.
5. Write results to a CSV under `results/`.

## Repository Structure

Key locations:

- `scripts/common/`
  The main runnable scripts.
- `src/reprodbench/pipeline/`
  Generator, refiner, and judge pipeline classes.
- `src/reprodbench/llm/prompts/`
  YAML prompt templates.
- `src/reprodbench/executor/sandbox_client.py`
  HTTP client used by reproduction scripts to execute generated code.
- `src/reprodbench/ablation/`
  Ablation modes and context builders.
- `data/`
  Input datasets currently included in the repo.

## Requirements

- Python `>=3.10,<4.0`
- One LLM backend:
  - Ollama for local models
  - OpenAI for OpenAI-hosted models
  - Anthropic for Anthropic-hosted models
- For reproduction scripts only: a sandbox execution service reachable through `SANDBOX_URL`

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

Most scripts call `load_dotenv()`, so create a `.env` file in the repo root.

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

What each one controls:

- `OPENAI_API_KEY`
  Required if a script uses `provider="openai"`.
- `ANTHROPIC_API_KEY`
  Required if a script uses `provider="anthropic"`.
- `SANDBOX_URL`
  Required for code reproduction scripts that execute generated code.
- `LOG_ENABLED`, `LOG_LEVEL`, `LOG_TO_FILE`, `LOG_FILE_PATH`
  Used by `src/reprodbench/utils/logger.py`.

## How Provider Selection Works

Provider selection is done inside each runner script, not from a global CLI.

The shared client in `src/reprodbench/llm/client.py` supports:

- `provider="ollama"`
- `provider="openai"`
- `provider="anthropic"`

The checked-in scripts currently use `provider="ollama"` almost everywhere.

If you want to switch providers, change the provider consistently in the same script for:

- the generator
- the refiner
- the judge

Example:

```python
provider="openai"
```

## Sandbox Requirement

These scripts need the sandbox service:

- `scripts/common/run_buggy_code_reproduction.py`
- `scripts/common/run_patched_code_reproduction.py`
- `scripts/common/run_patched_code_reproduction_no_answer.py`

They call `SandboxExecutorClient`, which sends requests to:

```text
{SANDBOX_URL}/execute
```

If the sandbox service is not running, extraction scripts can still work, but the reproduction scripts will fail when they try to execute code.

## Important Before You Run Anything

Some script defaults do not match the datasets currently checked into the repo.

For example, several scripts still reference:

- `data/run_github_issues_sampled.csv`
- placeholder mappings such as `path` or `path_to_guidance`

The datasets currently present in `data/` are:

- `data/reprodbench-base.csv`
- `data/reprodbench-gi.csv`
- `data/ReprodBench-verified.xlsx`

In practice, dataset paths usually need to be edited before a script will run successfully.

The checked-in scripts are reproducible, but not plug-and-play. Before running them, update the script-local config so the selected model names, dataset paths, and dataset columns match the files you actually have.

## Datasets Included In The Repo

### `data/reprodbench-base.csv`

This file starts with columns like:

- `question_id`
- `question`
- `answer`

### `data/reprodbench-gi.csv`

This file includes columns such as:

- `question_id`
- `question_link`
- `question_title`
- `question_body`
- `accepted_answer_body`
- `buggy_code`
- `patched_code`
- `python_version`
- `requirements`
- `repository`

This is the more useful starting point for patched reproduction workflows because it already contains runtime and code fields. It is not enough by itself for guided ablation runs, because the guidance columns still need to be added.

## How To Run The Project

There is no single top-level CLI. You run the scripts directly.

General form:

```bash
python scripts/common/<script_name>.py
```

Examples:

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

## Recommended Order

If you want to generate guidance first and then use it for reproduction:

### Buggy pipeline

1. Run `run_buggy_code_intent_extraction.py`
2. Run `run_buggy_fr_extraction.py`
3. Run `run_buggy_scot_extraction.py`
4. Merge those outputs into the dataset you want to use for code reproduction
5. Run `run_buggy_code_reproduction.py`

### Patched pipeline

1. Run `run_patched_code_intent_extraction.py`
2. Run `run_patched_functional_requirements_extraction.py`
3. Run `run_patched_scot_extraction.py`
4. Merge those outputs into the dataset you want to use for patched code generation
5. Run `run_patched_code_reproduction.py`

For exact reproducibility, use the same pattern for every model:

1. run the extraction scripts first
2. merge their outputs into a prepared CSV
3. point the reproduction script at that prepared CSV
4. run the reproduction script with the same model and ablation settings

## Where To Change What

This repo is configured mainly by editing constants near the top of each runner script.

### Change the generator model

For buggy extraction scripts, edit `GEN_MODELS`.

Files:

- `scripts/common/run_buggy_code_intent_extraction.py`
- `scripts/common/run_buggy_fr_extraction.py`
- `scripts/common/run_buggy_scot_extraction.py`
- `scripts/common/run_buggy_code_reproduction.py`

Example:

```python
GEN_MODELS = [
    "gpt-oss:20b",
]
```

For patched extraction and patched reproduction scripts, edit the model-to-dataset mapping:

- `MODEL_DATASETS`
- `MODEL_DATA_PATHS`

Example:

```python
MODEL_DATASETS = {
    "gpt-oss:20b": "data/your_input.csv",
}
```

```python
MODEL_DATA_PATHS = {
    "gpt-oss:20b": PROJECT_ROOT / "data" / "your_input.csv",
}
```

For reproduction scripts, also update `MODEL_SLUGS` when adding a new model name. Those scripts index `MODEL_SLUGS[MODEL_NAME]`, so adding the model only in `GEN_MODELS` or `MODEL_DATA_PATHS` is not sufficient.

Example:

```python
MODEL_SLUGS = {
    "gpt-oss:20b": "gpt-oss_20b",
}
```

### Change the judge model

Edit `JUDGE_MODEL_NAME` in the script you are running.

Example:

```python
JUDGE_MODEL_NAME = "qwen3:8b"
```

### Change the dataset path

This is the most common edit you will make.

Depending on the script, change one of:

- `DATA_PATH`
- `MODEL_DATASETS`
- `MODEL_DATA_PATHS`

Examples:

```python
DATA_PATH = PROJECT_ROOT / "data" / "reprodbench-gi.csv"
```

```python
MODEL_DATASETS = {
    "gpt-oss:20b": "data/prepared_guidance.csv",
}
```

```python
MODEL_DATA_PATHS = {
    "gpt-oss:20b": PROJECT_ROOT / "data" / "prepared_reproduction.csv",
}
```

### Change the ablation setup

Reproduction scripts use `ABLATIONS`.

Files:

- `scripts/common/run_buggy_code_reproduction.py`
- `scripts/common/run_patched_code_reproduction.py`
- `scripts/common/run_patched_code_reproduction_no_answer.py`

Example:

```python
ABLATIONS = [
    AblationMode.NONE,
    AblationMode.CI_FR_SCOT,
]
```

Common values are defined in `src/reprodbench/ablation/mode.py`.

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

### Change prompt templates

Prompt YAML files live under:

- `src/reprodbench/llm/prompts/thought_generation/buggy/`
- `src/reprodbench/llm/prompts/thought_generation/patched/`
- `src/reprodbench/llm/prompts/code_generation/buggy/`
- `src/reprodbench/llm/prompts/code_generation/patched/`
- `src/reprodbench/llm/prompts/judge_llm/buggy/`
- `src/reprodbench/llm/prompts/judge_llm/patched/`

If you want to change prompting behavior, edit the YAML file in the relevant directory.

### Change retry and timeout behavior

Common knobs in the runner scripts:

- `MAX_GENERATOR_RETRIES`
- `MAX_REFINER_RETRIES`
- `MAX_EXEC_RETRIES`
- `MAX_JUDGE_RETRIES`
- `MAX_CYCLES`
- `TIMEOUT_SECONDS`

### Change output locations

Each script defines its own `RESULTS_DIR`.

Common output folders include:

- `results/github/buggy_ci/`
- `results/github/buggy_fr/`
- `results/github/buggy_scot/`
- `results/github/buggy_code/`
- `results/github/patched_ci/`
- `results/github/ablation/patched_code/`
- `results/github/ablation/patched_code/no_answer/`

If you want results somewhere else, edit `RESULTS_DIR` in the target script.

## Script-By-Script Notes

### `run_buggy_code_intent_extraction.py`

Change:

- `GEN_MODELS`
- `JUDGE_MODEL_NAME`
- `DATA_PATH`

Run:

```bash
python scripts/common/run_buggy_code_intent_extraction.py
```

### `run_buggy_fr_extraction.py`

Change:

- `GEN_MODELS`
- `JUDGE_MODEL_NAME`
- `DATA_PATH`

Run:

```bash
python scripts/common/run_buggy_fr_extraction.py
```

### `run_buggy_scot_extraction.py`

Change:

- `GEN_MODELS`
- `JUDGE_MODEL_NAME`
- `DATA_PATH`

Run:

```bash
python scripts/common/run_buggy_scot_extraction.py
```

### `run_buggy_code_reproduction.py`

Change:

- `GEN_MODELS`
- `MODEL_DATA_PATHS`
- `MODEL_SLUGS`
- `JUDGE_MODEL_NAME`
- `ABLATIONS`
- `SANDBOX_URL` in `.env` if needed

Required input columns:

- `question_id`
- `question_title`
- `question_body`
- `question_link`
- `python_version`
- `requirements`

Guidance columns required by ablation:

- `AblationMode.NONE`
  No guidance columns required.
- `AblationMode.CI`
  `buggy_code_intent`
- `AblationMode.FR`
  `buggy_functional_requirements`
- `AblationMode.SCOT`
  `buggy_scot`
- `AblationMode.CI_FR`
  `buggy_code_intent`, `buggy_functional_requirements`
- `AblationMode.CI_SCOT`
  `buggy_code_intent`, `buggy_scot`
- `AblationMode.FR_SCOT`
  `buggy_functional_requirements`, `buggy_scot`
- `AblationMode.CI_FR_SCOT`
  `buggy_code_intent`, `buggy_functional_requirements`, `buggy_scot`

Minimal reproducible setup:

1. Run the three buggy extraction scripts.
2. Merge their outputs into one prepared CSV containing the required columns above.
3. Set `MODEL_DATA_PATHS[MODEL_NAME]` to that prepared CSV.
4. Make sure the same `MODEL_NAME` also exists in `MODEL_SLUGS`.
5. Start the sandbox service.
6. Run:

```bash
python scripts/common/run_buggy_code_reproduction.py
```

### `run_patched_code_intent_extraction.py`

Change:

- `MODEL_DATASETS`
- `JUDGE_MODEL_NAME`
- add a real CSV path in `MODEL_DATASETS`

Run:

```bash
python scripts/common/run_patched_code_intent_extraction.py
```

### `run_patched_functional_requirements_extraction.py`

Change:

- `MODEL_DATASETS`
- `JUDGE_MODEL_NAME`
- add a real CSV path in `MODEL_DATASETS`

Run:

```bash
python scripts/common/run_patched_functional_requirements_extraction.py
```

### `run_patched_scot_extraction.py`

Change:

- `MODEL_DATASETS`
- `JUDGE_MODEL_NAME`
- add a real CSV path in `MODEL_DATASETS`

Run:

```bash
python scripts/common/run_patched_scot_extraction.py
```

### `run_patched_code_reproduction.py`

Change:

- `MODEL_DATA_PATHS`
- `MODEL_SLUGS`
- `JUDGE_MODEL_NAME`
- `ABLATIONS`
- `SANDBOX_URL`

Required input columns:

- `question_id`
- `question_body`
- `accepted_answer_body`
- `buggy_code`
- `python_version`
- `requirements`

Guidance columns required by ablation:

- `AblationMode.NONE`
  No guidance columns required.
- `AblationMode.CI`
  `patched_code_intent`
- `AblationMode.FR`
  `patched_functional_requirements`
- `AblationMode.SCOT`
  `patched_scot`
- `AblationMode.CI_FR`
  `patched_code_intent`, `patched_functional_requirements`
- `AblationMode.CI_SCOT`
  `patched_code_intent`, `patched_scot`
- `AblationMode.FR_SCOT`
  `patched_functional_requirements`, `patched_scot`
- `AblationMode.CI_FR_SCOT`
  `patched_code_intent`, `patched_functional_requirements`, `patched_scot`

Minimal reproducible setup:

1. Run the three patched extraction scripts.
2. Merge their outputs into one prepared CSV containing the required columns above.
3. Set `MODEL_DATA_PATHS[MODEL_NAME]` to that prepared CSV.
4. Make sure the same `MODEL_NAME` also exists in `MODEL_SLUGS`.
5. Start the sandbox service.
6. Run:

```bash
python scripts/common/run_patched_code_reproduction.py
```

### `run_patched_code_reproduction_no_answer.py`

Change:

- `MODEL_DATA_PATHS`
- `MODEL_SLUGS`
- `JUDGE_MODEL_NAME`
- `ABLATIONS`
- `SANDBOX_URL`

This variant is for patched generation when you do not want the generator to use the accepted answer.

Required input columns:

- `question_id`
- `question_body`
- `buggy_code`
- `python_version`
- `requirements`

Optional but still used by the reviewer/judge path:

- `accepted_answer_body`

Guidance columns required by ablation:

- `patched_code_intent`
- `patched_functional_requirements`
- `patched_scot`

Only the columns implied by the selected `ABLATIONS` need to be present.

Run:

```bash
python scripts/common/run_patched_code_reproduction_no_answer.py
```

## Fastest Way To Get A First Run

1. Install the project.
2. Create `.env`.
3. Pick one script in `scripts/common/`.
4. Edit the model name, model mapping, and `MODEL_SLUGS` if it is a reproduction script.
5. Edit the dataset path in that script so it points to a file that actually has the required columns.
6. If it is a reproduction script, make sure the sandbox service is running.
7. Run the script with Python.

## Common Problems

### Dataset file not found

Cause:

- the script points to a dataset that is not checked into this repo
- the script still has a placeholder mapping such as `path`

Fix:

- update `DATA_PATH`, `MODEL_DATASETS`, or `MODEL_DATA_PATHS`

### CSV is missing columns

Cause:

- the dataset does not contain the fields the script reads

Fix:

- check the script’s main loop and align the input file with the columns it expects

### Ollama model not available

Cause:

- the script uses `provider="ollama"` but the model is not installed locally

Fix:

- pull the model in Ollama or switch the provider and model in the script

### Reproduction scripts fail during execution

Cause:

- the sandbox service is not running
- `SANDBOX_URL` is wrong
- generated code or requirements fail inside the sandbox

Fix:

- verify the sandbox service and the `.env` configuration

## Development Notes

Project configuration is in `pyproject.toml`.

Main dependencies include:

- `langchain`
- `langchain-openai`
- `langchain-anthropic`
- `langchain-ollama`
- `pandas`
- `python-dotenv`

Dev dependencies include:

- `pytest`
- `black`
- `isort`

## Summary

This project is not driven by a single command. The normal workflow is:

1. choose the script
2. edit the constants at the top of that script
3. make sure the dataset columns match the selected ablation
4. make sure `MODEL_SLUGS` also includes the chosen model for reproduction scripts
5. make sure the sandbox is running for reproduction scripts
6. run the script directly
