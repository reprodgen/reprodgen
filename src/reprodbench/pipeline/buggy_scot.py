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

BUGGY_SCOT_PATTERN = re.compile(
    r"<BUGGY_SCOT>\s*(.*?)\s*</BUGGY_SCOT>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# Artifact
# ============================================================


@dataclass
class BuggyScotArtifact:
    buggy_scot: str
    latency: float | None = None
    token_usage: dict | None = None


# ============================================================
# Generator
# ============================================================


class BuggyScotExtractor:
    """
    Extracts a buggy Structured Chain of Thought (SCoT) that reflects
    the user's procedural reasoning and control flow that leads to the
    buggy behavior described in the Stack Overflow question.

    The trace must reproduce the buggy outcome — not avoid it.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        loader = PromptLoader(prompt_dir)
        prompt_def = loader.load("buggy_scot_generator.yaml")

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
    ) -> BuggyScotArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = BUGGY_SCOT_PATTERN.search(raw)

        if not match:
            raise ValueError(f"<BUGGY_SCOT> block not found in LLM output:\n{raw}")

        return BuggyScotArtifact(
            buggy_scot=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )


# ============================================================
# Refiner (Judge-guided, string-based)
# ============================================================


class BuggyScotRefiner:
    """
    Refines a buggy SCoT using feedback from the buggy SCoT judge.

    Operates directly on the buggy_scot string to maintain
    clean generator–judge collaboration.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("buggy_scot_generator.yaml")

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
        buggy_scot: str,
        judge_label: str,
        judge_rationale: str,
    ) -> BuggyScotArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "buggy_scot": buggy_scot,
                "label": judge_label,
                "rationale": judge_rationale,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = BUGGY_SCOT_PATTERN.search(raw)

        if not match:
            raise ValueError(f"<BUGGY_SCOT> block not found in refined output:\n{raw}")

        return BuggyScotArtifact(
            buggy_scot=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )
