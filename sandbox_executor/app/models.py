from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class ExecuteRequest(BaseModel):
    code: str
    language: str = Field(default="python")
    python_version: str = Field(default="3.8")
    requirements: str = Field(default="")
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class ExecuteProjectRequest(BaseModel):
    files: Dict[str, str] = Field(
        ..., description="Mapping of relative file paths to file contents."
    )
    command: List[str] = Field(
        ...,
        description="Command argv to run inside the container, e.g. ['pytest','-q']",
    )
    python_version: str = Field(default="3.8")
    requirements: str = Field(default="")
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class ExecuteResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    runtime: int  # in milliseconds
    timeout: bool
    image: Optional[str]
    docker_available: bool
    error: Optional[str]
