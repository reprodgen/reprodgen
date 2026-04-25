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
from reprodbench.utils.text import preview
from reprodbench.utils.utils import strip_code_fences

# ============================================================
# Regex patterns (case-insensitive)
# ============================================================

PYTHON_VERSION_PATTERN = re.compile(
    r"<PYTHON_VERSION>\s*(.*?)\s*</PYTHON_VERSION>",
    re.DOTALL | re.IGNORECASE,
)
REQUIREMENT_PATTERN = re.compile(
    r"<REQUIREMENT>\s*(.*?)\s*</REQUIREMENT>",
    re.DOTALL | re.IGNORECASE,
)
PATCHED_CODE_PATTERN = re.compile(
    r"<PATCHED_CODE>\s*(.*?)\s*</PATCHED_CODE>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# Structured output error
# ============================================================


class StructuredOutputError(ValueError):
    def __init__(self, missing_tags: list[str], raw_text: str):
        self.missing_tags = missing_tags
        self.raw_text = raw_text
        super().__init__(
            "Missing required block(s): " + ", ".join(f"<{t}>" for t in missing_tags)
        )


def _try_extract(pattern: re.Pattern, text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = pattern.search(text)
    return match.group(1).strip() if match else None


# ============================================================
# Patched Semantic Context (Ablation input)
# ============================================================


@dataclass
class PatchedSemanticContext:
    patched_code_intent: str = ""
    patched_functional_requirements: str = ""
    patched_scot: str = ""


# ============================================================
# Artifact
# ============================================================


@dataclass
class PatchedCodeArtifact:
    python_version: str
    requirements: str
    patched_code: str
    latency: float | None = None
    token_usage: dict | None = None


# ============================================================
# Generator
# ============================================================


class PatchedCodeGenerator:
    """
    Generates patched (correct) code from:
      - original question
      - buggy code (for reference)
      - accepted Stack Overflow answer
      - optional patched semantics (CI / FR / SCOT)
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("patched_code_generator.yaml")

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

        self.chain = RunnableWithMessageHistory(
            prompt | llm,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        ).with_config(callbacks=[self.metrics])

    def generate(
        self,
        *,
        question: str,
        answer: str,
        buggy_code: str,
        python_version: str,
        requirements: str,
        semantic_context: PatchedSemanticContext,
        session_id: str,
    ) -> PatchedCodeArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
                "buggy_code": buggy_code,
                "python_version": python_version,
                "requirements": requirements,
                "patched_code_intent": semantic_context.patched_code_intent,
                "patched_functional_requirements": semantic_context.patched_functional_requirements,
                "patched_scot": semantic_context.patched_scot,
            },
            config={"configurable": {"session_id": session_id}},
        )

        artifact = self._parse(response.content)
        artifact.latency = self.metrics.latency
        artifact.token_usage = self.metrics.token_usage
        return artifact

    def _parse(self, raw_content) -> PatchedCodeArtifact:
        raw = str(raw_content).strip().replace("\r\n", "\n")

        python_version = _try_extract(PYTHON_VERSION_PATTERN, raw)
        requirements = _try_extract(REQUIREMENT_PATTERN, raw)
        patched_code = _try_extract(PATCHED_CODE_PATTERN, raw)

        missing = []
        if python_version is None:
            missing.append("PYTHON_VERSION")
        if requirements is None:
            missing.append("REQUIREMENT")
        if patched_code is None:
            missing.append("PATCHED_CODE")

        if missing:
            print("STRUCTURED OUTPUT VIOLATION (PATCHED GENERATOR)")
            print(raw[:800])
            raise StructuredOutputError(missing, raw)

        return PatchedCodeArtifact(
            python_version=python_version,
            requirements=requirements,
            patched_code=strip_code_fences(patched_code),
        )


# ============================================================
# Refiner
# ============================================================


class PatchedCodeRefiner:
    """
    Refines patched code when:
      - execution fails
      - judge reports mismatch
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("patched_code_generator.yaml")

        self.exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["refine_exec_error"]),
                MessagesPlaceholder("history"),
                ("human", "Revise and return ONLY the required tagged structure."),
            ]
        )

        self.judge_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["refine_judge_mismatch"]),
                MessagesPlaceholder("history"),
                ("human", "Revise and return ONLY the required tagged structure."),
            ]
        )

        llm = LLMClient(
            provider=provider,
            model_name=model,
            temperature=temperature,
        ).model

        self.exec_metrics = LLMRunMetrics()
        self.judge_metrics = LLMRunMetrics()

        # Use input_messages_key="input" since you pass {"input": "..."} and
        # want that to be the “new message” recorded in history.
        self.exec_chain = RunnableWithMessageHistory(
            self.exec_prompt | llm,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        ).with_config(callbacks=[self.exec_metrics])

        self.judge_chain = RunnableWithMessageHistory(
            self.judge_prompt | llm,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        ).with_config(callbacks=[self.judge_metrics])

    def _parse(
        self, raw_content, *, metrics: LLMRunMetrics | None = None
    ) -> PatchedCodeArtifact:
        raw = str(raw_content).strip().replace("\r\n", "\n")

        python_version = _try_extract(PYTHON_VERSION_PATTERN, raw)
        requirements = _try_extract(REQUIREMENT_PATTERN, raw)
        patched_code = _try_extract(PATCHED_CODE_PATTERN, raw)

        missing = []
        if python_version is None:
            missing.append("PYTHON_VERSION")
        if requirements is None:
            missing.append("REQUIREMENT")
        if patched_code is None:
            missing.append("PATCHED_CODE")

        if missing:
            print("STRUCTURED OUTPUT VIOLATION (PATCHED REFINER)")
            print(raw[:800])
            raise StructuredOutputError(missing, raw)

        artifact = PatchedCodeArtifact(
            python_version=python_version,
            requirements=requirements,
            patched_code=strip_code_fences(patched_code),
        )

        if metrics is not None:
            artifact.latency = metrics.latency
            artifact.token_usage = metrics.token_usage

        return artifact

    # --------------------------------------------------------
    # Execution-error refinement
    # --------------------------------------------------------

    def refine_exec_error(
        self,
        *,
        artifact: PatchedCodeArtifact,
        question: str,
        answer: str,
        stdout: str,
        stderr: str,
        docker_error: str,
        session_id: str,
        max_traceback_chars: int = 1200,
        max_docker_error_chars: int = 800,
    ) -> PatchedCodeArtifact:
        self.exec_metrics.reset()

        response = self.exec_chain.invoke(
            {
                "input": "refine_exec_error",
                "question": question,
                "answer": answer,
                "python_version": artifact.python_version,
                "requirements": artifact.requirements,
                "patched_code": artifact.patched_code,
                "stdout": stdout,
                "stderr": preview(stderr, max_traceback_chars),
                "docker_error": preview(docker_error, max_docker_error_chars),
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
        artifact: PatchedCodeArtifact,
        question: str,
        answer: str,
        stdout: str,
        stderr: str,
        judge_label: str,
        judge_rationale: str,
        session_id: str,
    ) -> PatchedCodeArtifact:
        self.judge_metrics.reset()

        response = self.judge_chain.invoke(
            {
                "input": "refine_judge_mismatch",
                "question": question,
                "answer": answer,
                "python_version": artifact.python_version,
                "requirements": artifact.requirements,
                "patched_code": artifact.patched_code,
                "stdout": stdout,
                "stderr": stderr,
                "judge_label": judge_label,
                "judge_rationale": judge_rationale,
            },
            config={"configurable": {"session_id": session_id}},
        )

        return self._parse(response.content, metrics=self.judge_metrics)
