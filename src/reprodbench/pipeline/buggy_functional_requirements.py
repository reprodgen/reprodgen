import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from reprodbench.llm.callbacks.metrics import LLMRunMetrics
from reprodbench.llm.client import LLMClient
from reprodbench.llm.prompts.prompt_loader import PromptLoader

# ============================================================
# Regex pattern (case-insensitive)
# ============================================================

BUGGY_FR_PATTERN = re.compile(
    r"<FUNCTIONAL_REQUIREMENTS>\s*(.*?)\s*</FUNCTIONAL_REQUIREMENTS>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# Artifact
# ============================================================


@dataclass
class BuggyFunctionalRequirementsArtifact:
    functional_requirements: str
    latency: float | None = None
    token_usage: dict | None = None


# ============================================================
# Generator
# ============================================================


class BuggyFunctionalRequirementsExtractor:
    """
    Extracts buggy functional requirements that reflect the user's
    mental model and intended behavior, grounded strictly in the
    Stack Overflow question (even if incorrect).
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load(
            "buggy_functional_requirements_generator.yaml"
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["system_prompt"]),
                ("human", prompt_def["task"]),
            ]
        )

        self.llm = LLMClient(
            provider=provider,
            model_name=model,
            temperature=temperature,
        )

        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | self.llm.model).with_config(
            callbacks=[self.metrics]
        )

    def extract(
        self,
        *,
        question: str,
    ) -> BuggyFunctionalRequirementsArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = BUGGY_FR_PATTERN.search(raw)

        if not match:
            raise ValueError(
                f"<FUNCTIONAL_REQUIREMENTS> block not found in LLM output:\n{raw}"
            )

        artifact = BuggyFunctionalRequirementsArtifact(
            functional_requirements=match.group(1).strip()
        )
        artifact.latency = self.metrics.latency
        artifact.token_usage = self.metrics.token_usage

        return artifact


# ============================================================
# Refiner (Judge-guided)
# ============================================================


class BuggyFunctionalRequirementsRefiner:
    """
    Refines buggy functional requirements using feedback from
    the buggy functional requirements judge.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load(
            "buggy_functional_requirements_generator.yaml"
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["system_prompt"]),
                ("human", prompt_def["refine"]),
            ]
        )

        self.llm = LLMClient(
            provider=provider,
            model_name=model,
            temperature=temperature,
        )

        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | self.llm.model).with_config(
            callbacks=[self.metrics]
        )

    def refine(
        self,
        *,
        question: str,
        functional_requirements: str,
        judge_label: str,
        judge_rationale: str,
    ) -> BuggyFunctionalRequirementsArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "functional_requirements": functional_requirements,
                "label": judge_label,
                "rationale": judge_rationale,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = BUGGY_FR_PATTERN.search(raw)

        if not match:
            raise ValueError(
                f"<FUNCTIONAL_REQUIREMENTS> block not found in refined output:\n{raw}"
            )

        artifact = BuggyFunctionalRequirementsArtifact(
            functional_requirements=match.group(1).strip()
        )
        artifact.latency = self.metrics.latency
        artifact.token_usage = self.metrics.token_usage

        return artifact
