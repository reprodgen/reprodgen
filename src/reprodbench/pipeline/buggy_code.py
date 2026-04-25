from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from reprodbench.llm.callbacks.metrics import LLMRunMetrics
from reprodbench.llm.client import LLMClient
from reprodbench.llm.memory import get_session_history
from reprodbench.llm.prompts.prompt_loader import PromptLoader
from reprodbench.utils.utils import debug_prompt, strip_code_fences

# ============================================================
# Regex patterns (case-insensitive)
# ============================================================

PYTHON_VERSION_PATTERN = re.compile(
    r"<PYTHON_VERSION>\s*(.*?)\s*</PYTHON_VERSION>", re.DOTALL | re.IGNORECASE
)
REQUIREMENT_PATTERN = re.compile(
    r"<REQUIREMENT>\s*(.*?)\s*</REQUIREMENT>", re.DOTALL | re.IGNORECASE
)
BUGGY_CODE_PATTERN = re.compile(
    r"<BUGGY_CODE>\s*(.*?)\s*</BUGGY_CODE>", re.DOTALL | re.IGNORECASE
)


# ============================================================
# Structured output error
# ============================================================


class StructuredOutputError(ValueError):
    """Raised when required XML-style blocks are missing from the LLM output."""

    def __init__(self, missing_tags: list[str], raw_text: str):
        self.missing_tags = missing_tags
        self.raw_text = raw_text
        super().__init__(
            "Missing required block(s): " + ", ".join(f"<{t}>" for t in missing_tags)
        )


def _try_extract(pattern: re.Pattern, text) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = pattern.search(text)
    return match.group(1).strip() if match else None


# ============================================================
# Semantic Context (Ablation inputs)
# ============================================================


@dataclass
class SemanticContext:
    buggy_code_intent: str = ""
    buggy_functional_requirements: str = ""
    buggy_scot: str = ""


# ============================================================
# Artifact
# ============================================================


@dataclass
class BuggyCodeArtifact:
    python_version: str
    requirements: str
    buggy_code: str
    latency: float | None = None
    token_usage: dict | None = None


# ============================================================
# Generator
# ============================================================


class BuggyCodeGenerator:
    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("buggy_code_generator.yaml")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["system_prompt"]),
                MessagesPlaceholder("history"),
                ("human", prompt_def["task"]),
            ]
        )

        llm = LLMClient(
            provider=provider,
            model_name=model,
            temperature=temperature,
        ).model

        self.metrics = LLMRunMetrics()

        # IMPORTANT: callbacks should be attached to the runnable (not just invoke-config)
        self.chain = RunnableWithMessageHistory(
            prompt | debug_prompt | llm,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        ).with_config(callbacks=[self.metrics])

    def generate(
        self,
        *,
        question_text: str,
        semantic_context: SemanticContext,
        session_id: str,
    ) -> BuggyCodeArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question_text,
                "buggy_code_intent": semantic_context.buggy_code_intent,
                "buggy_functional_requirements": semantic_context.buggy_functional_requirements,
                "buggy_scot": semantic_context.buggy_scot,
            },
            config={"configurable": {"session_id": session_id}},
        )

        raw = str(response.content).strip().replace("\r\n", "\n")

        python_version = _try_extract(PYTHON_VERSION_PATTERN, raw)
        requirements = _try_extract(REQUIREMENT_PATTERN, raw)
        buggy_code = _try_extract(BUGGY_CODE_PATTERN, raw)

        missing: list[str] = []
        if python_version is None:
            missing.append("PYTHON_VERSION")
        if requirements is None:
            missing.append("REQUIREMENT")
        if buggy_code is None:
            missing.append("BUGGY_CODE")

        if missing:
            raise StructuredOutputError(missing, raw)

        artifact = BuggyCodeArtifact(
            python_version=python_version,
            requirements=requirements,
            buggy_code=strip_code_fences(buggy_code),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )
        return artifact


# ============================================================
# Refiner
# ============================================================


class BuggyCodeRefiner:
    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("buggy_code_generator.yaml")

        self.exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["refine_exec_error"]),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "Revise the artifact and return ONLY the required tagged structure.",
                ),
            ]
        )

        self.judge_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["refine_judge_mismatch"]),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "Revise the artifact and return ONLY the required tagged structure.",
                ),
            ]
        )

        llm = LLMClient(
            provider=provider,
            model_name=model,
            temperature=temperature,
        ).model

        self.exec_metrics = LLMRunMetrics()
        self.judge_metrics = LLMRunMetrics()

        self.exec_chain = RunnableWithMessageHistory(
            debug_prompt | self.exec_prompt | llm,
            get_session_history,
            input_messages_key=None,
            history_messages_key="history",
        ).with_config(callbacks=[self.exec_metrics])

        self.judge_chain = RunnableWithMessageHistory(
            debug_prompt | self.judge_prompt | llm,
            get_session_history,
            input_messages_key=None,
            history_messages_key="history",
        ).with_config(callbacks=[self.judge_metrics])

    # --------------------------------------------------------
    # Parse helper (attaches metrics when provided)
    # --------------------------------------------------------

    def _parse(
        self,
        raw_content,
        *,
        metrics: LLMRunMetrics | None = None,
    ) -> BuggyCodeArtifact:
        raw = str(raw_content).strip().replace("\r\n", "\n")

        python_version = _try_extract(PYTHON_VERSION_PATTERN, raw)
        requirements = _try_extract(REQUIREMENT_PATTERN, raw)
        buggy_code = _try_extract(BUGGY_CODE_PATTERN, raw)

        missing: list[str] = []
        if python_version is None:
            missing.append("PYTHON_VERSION")
        if requirements is None:
            missing.append("REQUIREMENT")
        if buggy_code is None:
            missing.append("BUGGY_CODE")

        if missing:
            raise StructuredOutputError(missing, raw)

        return BuggyCodeArtifact(
            python_version=python_version,
            requirements=requirements,
            buggy_code=strip_code_fences(buggy_code),
            latency=(metrics.latency if metrics is not None else None),
            token_usage=(metrics.token_usage if metrics is not None else None),
        )

    # --------------------------------------------------------
    # Execution-error refinement
    # --------------------------------------------------------

    def refine(
        self,
        *,
        artifact: BuggyCodeArtifact,
        buggy_stderr: str,
        docker_error: str,
        session_id: str,
    ) -> BuggyCodeArtifact:
        self.exec_metrics.reset()

        response = self.exec_chain.invoke(
            {
                "input": "refine_exec_error",
                "python_version": artifact.python_version,
                "requirements": artifact.requirements,
                "buggy_code": artifact.buggy_code,
                "buggy_stderr": buggy_stderr,
                "docker_error": docker_error,
            },
            config={"configurable": {"session_id": session_id}},
        )

        return self._parse(response.content, metrics=self.exec_metrics)

    # --------------------------------------------------------
    # Judge-mismatch refinement
    # --------------------------------------------------------

    def refine_judge_mismatch(
        self,
        *,
        artifact: BuggyCodeArtifact,
        question: str,
        stdout: str,
        stderr: str,
        judge_label: str,
        judge_rationale: str,
        session_id: str,
    ) -> BuggyCodeArtifact:
        self.judge_metrics.reset()

        response = self.judge_chain.invoke(
            {
                "input": "refine_judge_mismatch",
                "question": question,
                "python_version": artifact.python_version,
                "requirements": artifact.requirements,
                "buggy_code": artifact.buggy_code,
                "stdout": stdout,
                "stderr": stderr,
                "judge_label": judge_label,
                "judge_rationale": judge_rationale,
            },
            config={"configurable": {"session_id": session_id}},
        )

        return self._parse(response.content, metrics=self.judge_metrics)
