import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from reprodbench.llm.callbacks.metrics import LLMRunMetrics
from reprodbench.llm.client import LLMClient
from reprodbench.llm.prompts.prompt_loader import PromptLoader

# ============================================================
# Regex patterns (case-insensitive)
# ============================================================

PATCHED_INTENT_PATTERN = re.compile(
    r"<PATCHED_INTENT>\s*(.*?)\s*</PATCHED_INTENT>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# Artifact
# ============================================================


@dataclass
class PatchedCodeIntentArtifact:
    patched_code_intent: str
    latency: Optional[float] = None
    token_usage: Optional[Dict[str, Any]] = None


# ============================================================
# Generator
# ============================================================


class PatchedCodeIntentExtractor:
    """
    Extracts the patched code intent from a Stack Overflow question
    and its accepted answer.

    The intent describes the functional goal of the corrected code,
    abstracted from implementation details.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("patched_code_intent_generator.yaml")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["system_prompt"]),
                ("human", prompt_def["task"]),
            ]
        )

        llm = LLMClient(
            provider=provider,
            model_name=model,
            temperature=temperature,
        ).model

        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | llm).with_config(callbacks=[self.metrics])

    # --------------------------------------------------------

    def extract(
        self,
        *,
        question: str,
        answer: str,
    ) -> PatchedCodeIntentArtifact:

        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = PATCHED_INTENT_PATTERN.search(raw)

        if not match:
            raise ValueError(
                f"<PATCHED_INTENT> block not found in extractor output:\n{raw}"
            )

        return PatchedCodeIntentArtifact(
            patched_code_intent=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )


# ============================================================
# Refiner (Judge-guided)
# ============================================================


class PatchedCodeIntentRefiner:
    """
    Refines patched code intent using feedback from the
    patched code intent judge.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("patched_code_intent_generator.yaml")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["system_prompt"]),
                ("human", prompt_def["refine"]),
            ]
        )

        llm = LLMClient(
            provider=provider,
            model_name=model,
            temperature=temperature,
        ).model

        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | llm).with_config(callbacks=[self.metrics])

    # --------------------------------------------------------

    def refine(
        self,
        *,
        question: str,
        answer: str,
        patched_code_intent: str,
        judge_label: str,
        judge_rationale: str,
    ) -> PatchedCodeIntentArtifact:

        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
                "patched_code_intent": patched_code_intent,
                "label": judge_label,
                "rationale": judge_rationale,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = PATCHED_INTENT_PATTERN.search(raw)

        if not match:
            raise ValueError(
                f"<PATCHED_INTENT> block not found in refiner output:\n{raw}"
            )

        return PatchedCodeIntentArtifact(
            patched_code_intent=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )
