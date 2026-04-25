import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from reprodbench.llm.callbacks.metrics import LLMRunMetrics
from reprodbench.llm.client import LLMClient
from reprodbench.llm.prompts.prompt_loader import PromptLoader

# ============================================================
# Regex patterns (case-insensitive)
# ============================================================

LABEL_PATTERN = re.compile(
    r"<\s*LABEL\s*>\s*(.*?)\s*<\s*/\s*LABEL\s*>",
    re.DOTALL | re.IGNORECASE,
)

RATIONALE_PATTERN = re.compile(
    r"<\s*RATIONALE\s*>\s*(.*?)\s*<\s*/\s*RATIONALE\s*>",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# Judge Result Artifact
# ============================================================


@dataclass
class JudgeResult:
    label: str
    rationale: str

    # --- Metrics ---
    latency: Optional[float] = None
    token_usage: Optional[dict] = None
    model_name: Optional[str] = None


# ============================================================
# Judge
# ============================================================


class BuggyCodeJudge:
    """
    LLM-based judge for buggy code reproduction.

    Produces:
      - correctness label
      - rationale
      - latency
      - token usage
      - model name
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        loader = PromptLoader(prompt_dir)
        prompt_def = loader.load("judge_buggy_code_semantic.yaml")

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

        # ---- Metrics callback ----
        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | self.llm).with_config(callbacks=[self.metrics])

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def judge(
        self,
        *,
        question: str,
        buggy_code: str,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> JudgeResult:
        """
        Invoke the judge LLM and return a JudgeResult artifact.
        """
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "buggy_code": buggy_code,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }
        )

        raw = str(response.content).strip()

        label = self._extract(LABEL_PATTERN, raw, "LABEL").lower()
        rationale = self._extract(RATIONALE_PATTERN, raw, "RATIONALE")

        return JudgeResult(
            label=label,
            rationale=rationale,
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
            model_name=self.metrics.model_name,
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _extract(pattern: re.Pattern, text: str, tag: str) -> str:
        match = pattern.search(text)
        if not match:
            raise ValueError(f"<{tag}> block not found in judge LLM output:\n{text}")
        return match.group(1).strip()
