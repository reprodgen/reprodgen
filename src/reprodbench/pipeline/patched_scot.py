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

PATCHED_SCOT_PATTERN = re.compile(
    r"<PATCHED_SCOT>\s*(.*?)\s*</PATCHED_SCOT>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# Artifact
# ============================================================


@dataclass
class PatchedScotArtifact:
    patched_scot: str
    latency: float | None = None
    token_usage: dict | None = None


# ============================================================
# Generator
# ============================================================


class PatchedScotExtractor:
    """
    Extracts a patched Structured Chain of Thought (SCoT) that reflects
    the procedural reasoning and control flow of the corrected code,
    grounded strictly in the accepted Stack Overflow answer.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        loader = PromptLoader(prompt_dir)
        prompt_def = loader.load("patched_scot_generator.yaml")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["system_prompt"]),
                ("human", prompt_def["task"]),
            ]
        )

        # IMPORTANT: use `.model` for LangChain runnable
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
    ) -> PatchedScotArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = PATCHED_SCOT_PATTERN.search(raw)

        if not match:
            raise ValueError(f"<PATCHED_SCOT> block not found in LLM output:\n{raw}")

        return PatchedScotArtifact(
            patched_scot=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )


# ============================================================
# Refiner (Judge-guided)
# ============================================================


class PatchedScotRefiner:
    """
    Refines a patched SCoT using feedback from the patched SCoT judge.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("patched_scot_generator.yaml")

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
        patched_scot: str,
        judge_label: str,
        judge_rationale: str,
    ) -> PatchedScotArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
                "patched_scot": patched_scot,
                "label": judge_label,
                "rationale": judge_rationale,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = PATCHED_SCOT_PATTERN.search(raw)

        if not match:
            raise ValueError(
                f"<PATCHED_SCOT> block not found in refined output:\n{raw}"
            )

        return PatchedScotArtifact(
            patched_scot=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )
