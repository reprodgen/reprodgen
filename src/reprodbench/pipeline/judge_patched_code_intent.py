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

LABEL_PATTERN = re.compile(
    r"<\s*LABEL\s*>\s*(.*?)\s*<\s*/\s*LABEL\s*>",
    re.DOTALL | re.IGNORECASE,
)

RATIONALE_PATTERN = re.compile(
    r"<\s*RATIONALE\s*>\s*(.*?)\s*<\s*/\s*RATIONALE\s*>",
    re.DOTALL | re.IGNORECASE,
)

# ============================================================
# Artifact
# ============================================================


@dataclass
class PatchedIntentJudgeResult:
    label: str
    rationale: str
    latency: Optional[float] = None
    token_usage: Optional[Dict[str, Any]] = None


# ============================================================
# Judge
# ============================================================


class PatchedCodeIntentJudge:
    """
    Judges whether a generated patched code intent accurately captures
    the functional goal of the corrected code as described in the
    accepted Stack Overflow answer.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("judge_patched_code_intent.yaml")

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

        # metrics callback
        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | llm).with_config(callbacks=[self.metrics])

    # --------------------------------------------------------

    def judge(
        self,
        *,
        question: str,
        answer: str,
        patched_code_intent: str,
    ) -> PatchedIntentJudgeResult:

        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
                "patched_code_intent": patched_code_intent,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")

        label = self._extract(LABEL_PATTERN, raw, "LABEL")
        rationale = self._extract(RATIONALE_PATTERN, raw, "RATIONALE")

        return PatchedIntentJudgeResult(
            label=label.lower(),
            rationale=rationale,
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )

    # --------------------------------------------------------

    @staticmethod
    def _extract(pattern: re.Pattern, text: str, tag: str) -> str:
        match = pattern.search(text)
        if not match:
            raise ValueError(
                f"<{tag}> block not found in patched intent judge output:\n{text}"
            )
        return match.group(1).strip()
