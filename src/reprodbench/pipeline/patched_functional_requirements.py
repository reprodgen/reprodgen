import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from reprodbench.llm.callbacks.metrics import LLMRunMetrics
from reprodbench.llm.client import LLMClient
from reprodbench.llm.prompts.prompt_loader import PromptLoader

# ============================================================
# Regex pattern
# ============================================================

PATCHED_FR_PATTERN = re.compile(
    r"<FUNCTIONAL_REQUIREMENTS>\s*(.*?)\s*</FUNCTIONAL_REQUIREMENTS>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# Artifact
# ============================================================


@dataclass
class PatchedFunctionalRequirementsArtifact:
    functional_requirements: str
    latency: float | None = None
    token_usage: dict | None = None


# ============================================================
# Generator
# ============================================================


class PatchedFunctionalRequirementsExtractor:
    """
    Extracts patched functional requirements that reflect the
    corrected behavior of the code, grounded strictly in the
    accepted Stack Overflow answer.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        loader = PromptLoader(prompt_dir)
        prompt_def = loader.load("patched_functional_requirements_generator.yaml")

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
        ).model

        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | self.llm).with_config(callbacks=[self.metrics])

    def extract(
        self,
        *,
        question: str,
        answer: str,
    ) -> PatchedFunctionalRequirementsArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = PATCHED_FR_PATTERN.search(raw)

        if not match:
            raise ValueError(
                f"<FUNCTIONAL_REQUIREMENTS> block not found in LLM output:\n{raw}"
            )

        return PatchedFunctionalRequirementsArtifact(
            functional_requirements=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )


# ============================================================
# Refiner (Judge-guided)
# ============================================================


class PatchedFunctionalRequirementsRefiner:
    """
    Refines patched functional requirements using feedback from
    the patched functional requirements judge.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load(
            "patched_functional_requirements_generator.yaml"
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
        ).model

        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | self.llm).with_config(callbacks=[self.metrics])

    def refine(
        self,
        *,
        question: str,
        answer: str,
        functional_requirements: str,
        judge_label: str,
        judge_rationale: str,
    ) -> PatchedFunctionalRequirementsArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
                "functional_requirements": functional_requirements,
                "label": judge_label,
                "rationale": judge_rationale,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = PATCHED_FR_PATTERN.search(raw)

        if not match:
            raise ValueError(
                f"<FUNCTIONAL_REQUIREMENTS> block not found in refined output:\n{raw}"
            )

        return PatchedFunctionalRequirementsArtifact(
            functional_requirements=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )
