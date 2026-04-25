# reprodbench/ablation/context.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class BuggySemanticContext:
    buggy_code_intent: Optional[str] = None
    buggy_functional_requirements: Optional[str] = None
    buggy_scot: Optional[str] = None


@dataclass
class PatchedSemanticContext:
    patched_code_intent: Optional[str] = None
    patched_functional_requirements: Optional[str] = None
    patched_scot: Optional[str] = None
