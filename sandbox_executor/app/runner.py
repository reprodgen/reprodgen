import subprocess
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, List

from app.execution_environment import (
    ensure_environment,
    check_docker_available,
    EnvironmentBuildError,
)


# ----------------------------------------------------------------------
# Environment preparation
# ----------------------------------------------------------------------

def prepare_execution_environment(
    python_version: str,
    requirements: str,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Returns:
        (image_tag, None) on success
        (None, error_response_dict) on failure
    """
    if not check_docker_available():
        return None, {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "runtime": 0,
            "timeout": False,
            "image": None,
            "docker_available": False,
            "error": "Docker daemon is not running",
        }

    try:
        image_tag = ensure_environment(python_version, requirements)
        return image_tag, None

    except EnvironmentBuildError as e:
        return None, {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "runtime": 0,
            "timeout": False,
            "image": None,
            "docker_available": True,
            "error": str(e),
        }


# ----------------------------------------------------------------------
# Single-file execution
# ----------------------------------------------------------------------

def run_python_code(
    code: str,
    requirements: str,
    python_version: str,
    timeout_seconds: int,
) -> dict:

    image_tag, error = prepare_execution_environment(python_version, requirements)
    if error:
        return error

    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / "code.py"
        code_path.write_text(code)

        try:
            result = subprocess.run(
                [
                    "docker", "run",
                    "--rm",
                    "--network", "none",
                    "--cpus", "10",
                    "--memory", "10g",
                    "--pids-limit", "64",
                    "-v", f"{code_path}:/exec/code.py:ro",
                    image_tag,
                    "python", "/exec/code.py",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            runtime = int((time.time() - start_time) * 1000)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "runtime": runtime,
                "timeout": False,
                "image": image_tag,
                "docker_available": True,
                "error": None,
            }

        except subprocess.TimeoutExpired:
            runtime = int((time.time() - start_time) * 1000)
            return {
                "stdout": "",
                "stderr": "Execution timed out",
                "exit_code": -1,
                "runtime": runtime,
                "timeout": True,
                "image": image_tag,
                "docker_available": True,
                "error": None,
            }


# ----------------------------------------------------------------------
# Multi-file / project execution
# ----------------------------------------------------------------------

def run_python_project(
    files: Dict[str, str],
    command: List[str],
    requirements: str,
    python_version: str,
    timeout_seconds: int,
) -> dict:

    image_tag, error = prepare_execution_environment(python_version, requirements)
    if error:
        return error

    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write project files safely
        for filename, content in files.items():
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts:
                return {
                    "stdout": "",
                    "stderr": f"Invalid file path: {filename}",
                    "exit_code": -1,
                    "runtime": 0,
                    "timeout": False,
                    "image": image_tag,
                    "docker_available": True,
                    "error": f"Invalid file path: {filename}",
                }

            file_path = tmp / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        docker_command = [
            "docker", "run",
            "--rm",
            "--network", "none",
            "--cpus", "10",
            "--memory", "10g",
            "--pids-limit", "64",
            "-v", f"{tmp}:/exec",
            "-w", "/exec",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", "PYTHONPATH=/exec",
            "-e", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            image_tag,
            *command,
        ]

        try:
            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            runtime = int((time.time() - start_time) * 1000)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "runtime": runtime,
                "timeout": False,
                "image": image_tag,
                "docker_available": True,
                "error": None,
            }

        except subprocess.TimeoutExpired:
            runtime = int((time.time() - start_time) * 1000)
            return {
                "stdout": "",
                "stderr": "Execution timed out",
                "exit_code": -1,
                "runtime": runtime,
                "timeout": True,
                "image": image_tag,
                "docker_available": True,
                "error": None,
            }
