# src/reprodbench/llm/context/types.py
from dataclasses import dataclass
from enum import Enum


class ContextSource(str, Enum):
    QUESTION = "question"
    BUGGY_CODE = "buggy_code"
    STDOUT = "stdout"
    STDERR = "stderr"
    DOCKER_ERROR = "docker_error"
    JUDGE_FEEDBACK = "judge_feedback"


@dataclass
class ContextItem:
    source: ContextSource
    content: str
    priority: int  # lower = more important
