from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    runtime: Optional[int]
    timeout: bool
    image: Optional[str]
    docker_available: bool
    error: Optional[str]


class SandboxExecutorClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def execute_python(
        self,
        code: str,
        requirements: str,
        python_version: str,
        timeout_seconds: int = 20,
    ) -> ExecutionResult:
        payload = {
            "language": "python",
            "code": code,
            "requirements": requirements,
            "python_version": python_version,
            "timeout_seconds": timeout_seconds,
        }

        resp = requests.post(f"{self.base_url}/execute", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return ExecutionResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exit_code", -1),
            runtime=data.get("runtime"),
            timeout=data.get("timeout", False),
            image=data.get("image"),
            docker_available=data.get("docker_available", True),
            error=data.get("error"),
        )
