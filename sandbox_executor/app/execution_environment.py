import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from filelock import FileLock


class EnvironmentBuildError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Docker availability
# ----------------------------------------------------------------------

def check_docker_available() -> bool:
    """
    Check whether Docker daemon is available and reachable.
    """
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ----------------------------------------------------------------------
# Version normalization utilities
# ----------------------------------------------------------------------

def normalize_python_version(python_version: Optional[str]) -> str:
    """
    Convert LLM-produced python version strings into a normalized numeric form.

    Examples:
      "python 3.9"   -> "39"
      "Python3.10"   -> "310"
      "3.11"         -> "311"
      None / ""      -> "310"
    """
    if not python_version:
        return "310"

    pv = python_version.lower().strip()
    pv = pv.replace("python", "").strip()
    pv = pv.replace(".", "")
    pv = "".join(c for c in pv if c.isdigit())

    return pv or "310"


def python_base_image_version(normalized_version: str) -> str:
    """
    Convert normalized version like '39', '310', '311'
    into Docker base image versions: '3.9', '3.10', '3.11'
    """
    if len(normalized_version) == 2:
        return f"{normalized_version[0]}.{normalized_version[1]}"
    return f"{normalized_version[0]}.{normalized_version[1:]}"


# ----------------------------------------------------------------------
# Image tag computation
# ----------------------------------------------------------------------

def compute_env_tag(normalized_version: str, requirements: str) -> str:
    """
    Compute a stable, Docker-safe image tag based on
    python version + requirements content.
    """
    key = f"{normalized_version}\n{requirements}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"sandbox_py{normalized_version}_{digest}"


def _image_exists(image_tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


# ----------------------------------------------------------------------
# Environment builder (PARALLEL SAFE)
# ----------------------------------------------------------------------

def ensure_environment(python_version: Optional[str], requirements: str) -> str:
    """
    Ensure a Docker image exists for the given Python version
    and requirements. Build it if missing.

    PARALLEL-SAFE.
    """
    normalized_version = normalize_python_version(python_version)
    base_version = python_base_image_version(normalized_version)
    image_tag = compute_env_tag(normalized_version, requirements)

    lock_path = Path("/tmp") / f"docker-env-{image_tag}.lock"
    lock = FileLock(str(lock_path))

    with lock:
        if _image_exists(image_tag):
            return image_tag

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            (tmp / "requirements.txt").write_text(requirements or "")

            (tmp / "Dockerfile").write_text(
                f"""
FROM python:{base_version}-slim
WORKDIR /exec
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \\
 && pip install --no-cache-dir -r requirements.txt
"""
            )

            try:
                subprocess.run(
                    ["docker", "build", "-t", image_tag, tmpdir],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise EnvironmentBuildError(
                    f"Environment build failed.\n"
                    f"Base image: python:{base_version}-slim\n"
                    f"Requirements:\n{requirements}\n\n"
                    f"Build stderr:\n{e.stderr}"
                )

    return image_tag
