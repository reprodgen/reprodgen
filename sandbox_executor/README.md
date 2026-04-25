# Sandbox Executor

`sandbox_executor` is a small FastAPI service that runs generated Python code inside Docker and returns the process output. ReprodBench uses it for reproduction scripts that need to execute model-generated code without running that code directly on the host.

## What It Supports

- Single-file Python execution through `POST /execute`
- Multi-file project execution through `POST /execute_project`
- Per-request Python version selection
- Per-request dependency installation from a `requirements.txt` string
- Cached Docker images keyed by Python version and requirements content

## How It Works

For each request, the service:

1. Checks whether Docker is available.
2. Normalizes the requested Python version.
3. Builds or reuses a Docker image for that Python version plus the supplied requirements.
4. Writes the submitted code or project files into a temporary directory.
5. Runs the code in a Docker container with networking disabled.
6. Returns `stdout`, `stderr`, exit code, runtime, and timeout metadata.

Environment images are cached by a stable tag derived from:

- normalized Python version
- exact `requirements` string

That means repeated requests with the same runtime configuration reuse the same image instead of reinstalling dependencies every time.

## Runtime Isolation

Execution currently uses these Docker flags:

- `--network none`
- `--cpus 10`
- `--memory 10g`
- `--pids-limit 64`

For project execution, the service also sets:

- `PYTHONDONTWRITEBYTECODE=1`
- `PYTHONUNBUFFERED=1`
- `PYTHONPATH=/exec`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`

This is a practical isolation layer for benchmark execution, not a hardened multi-tenant sandbox.

## Requirements

- Python `>=3.9`
- Docker installed and the Docker daemon running
- A machine that can build Docker images such as `python:3.8-slim`, `python:3.10-slim`, or similar depending on the requested version

Python dependencies for the API itself are defined in [pyproject.toml](/home/ipd21/Documents/reprodgen-final/sandbox_executor/pyproject.toml).

## Install

From [sandbox_executor](/home/ipd21/Documents/reprodgen-final/sandbox_executor):

```bash
cd sandbox_executor
uv sync
```

Or with `pip`:

```bash
cd sandbox_executor
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Start The Service

Start the API from [sandbox_executor](/home/ipd21/Documents/reprodgen-final/sandbox_executor):

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

By default, ReprodBench expects:

```env
SANDBOX_URL=http://localhost:8000
```

## How To Use It

Typical flow:

1. Start Docker.
2. Install the sandbox executor dependencies.
3. Launch the API with `uvicorn`.
4. Send requests to `POST /execute` or `POST /execute_project`.
5. Point the main repo at the service with `SANDBOX_URL`.

## Quick Start

From the repo root, you can bring up the service with:

```bash
cd sandbox_executor
uv sync
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

In another shell, test it with:

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

Expected result:

- `stdout` contains `sandbox ok`
- `exit_code` is `0`
- `docker_available` is `true`

If this request fails, fix Docker or dependency installation before trying to run ReprodGen reproduction scripts.

## API

### `POST /execute`

Runs a single Python source string as `/exec/code.py`.

Request body:

```json
{
  "code": "print('hello')",
  "language": "python",
  "python_version": "3.10",
  "requirements": "",
  "timeout_seconds": 30
}
```

Notes:

- `language` must be `"python"` or the API returns `400`
- `timeout_seconds` must be between `1` and `300`
- `python_version` defaults to `"3.8"` in the request model, but empty or missing values are normalized internally to Python `3.10`

Example:

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import sys\nprint(sys.version)",
    "language": "python",
    "python_version": "3.10",
    "requirements": "",
    "timeout_seconds": 30
  }'
```

### `POST /execute_project`

Runs a multi-file project by mounting submitted files into `/exec` and executing the provided argv command.

Request body:

```json
{
  "files": {
    "math_utils.py": "def add(a, b):\n    return a + b\n",
    "test_math_utils.py": "from math_utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
  },
  "command": ["pytest", "-q"],
  "python_version": "3.10",
  "requirements": "pytest==8.3.5",
  "timeout_seconds": 30
}
```

Notes:

- `files` must use relative paths
- absolute paths and any path containing `..` are rejected
- `command` is passed directly to Docker as argv, for example `["python", "main.py"]` or `["pytest", "-q"]`

Example:

```bash
curl -X POST http://localhost:8000/execute_project \
  -H "Content-Type: application/json" \
  -d '{
    "files": {
      "main.py": "print(\"ok\")\n"
    },
    "command": ["python", "main.py"],
    "python_version": "3.10",
    "requirements": "",
    "timeout_seconds": 30
  }'
```

## Response Shape

Both endpoints return:

```json
{
  "stdout": "ok\n",
  "stderr": "",
  "exit_code": 0,
  "runtime": 123,
  "timeout": false,
  "image": "sandbox_py310_abcd1234ef567890",
  "docker_available": true,
  "error": null
}
```

Fields:

- `stdout`: captured standard output
- `stderr`: captured standard error
- `exit_code`: process exit status, or `-1` on timeout/setup failure
- `runtime`: total request execution time in milliseconds
- `timeout`: whether the subprocess hit the request timeout
- `image`: Docker image tag used for execution
- `docker_available`: whether Docker was reachable by the service
- `error`: setup/build error details, if any

## Python Version Normalization

The service accepts flexible Python version strings and normalizes them before selecting a base image.

Examples:

- `"python 3.9"` -> `python:3.9-slim`
- `"Python3.10"` -> `python:3.10-slim`
- `"3.11"` -> `python:3.11-slim`
- `""` or `null` -> `python:3.10-slim`

## Failure Modes

Common failures include:

- Docker is not installed or the daemon is not running
- dependency installation fails while building the environment image
- submitted code exits non-zero
- execution times out
- project files include invalid paths

If dependency installation fails, `error` contains the Docker build stderr along with the selected base image and submitted requirements.

## Integration With ReprodBench

The main repository uses this service through `src/reprodbench/executor/sandbox_client.py`. Reproduction scripts call:

- `POST {SANDBOX_URL}/execute`

Typical end-to-end usage from the main repo:

1. Start the service from `sandbox_executor/`.
2. Set `SANDBOX_URL=http://localhost:8000` in the repo root `.env`.
3. Prepare the reproduction dataset expected by the script you want to run.
4. Run a reproduction script such as `python scripts/common/run_buggy_code_reproduction.py`.

Set `SANDBOX_URL` in the repo root `.env` file before running those scripts:

```env
SANDBOX_URL=http://localhost:8000
```

If the service is down, extraction-only workflows can still run, but code reproduction workflows that require execution will fail.
