import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from reprodbench.llm.callbacks.metrics import LLMRunMetrics
from reprodbench.llm.client import LLMClient
from reprodbench.llm.prompts.prompt_loader import PromptLoader

# ============================================================
# Regex patterns
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
class JudgeResult:
    label: str
    rationale: str
    latency: float | None = None
    token_usage: dict | None = None


# ============================================================
# Judge
# ============================================================


class PatchedScotJudge:
    """
    Judges whether a generated patched Structured Chain of Thought (SCoT)
    accurately reflects the procedural reasoning and control structure
    of the corrected code as described in the accepted Stack Overflow answer.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        loader = PromptLoader(prompt_dir)
        prompt_def = loader.load("judge_patched_scot.yaml")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_def["system_prompt"]),
                ("human", prompt_def["task"]),
            ]
        )

        # IMPORTANT:
        # - use `.model` (LangChain Runnable)
        # - DO NOT access client-level metrics
        self.llm = LLMClient(
            provider=provider,
            model_name=model,
            temperature=temperature,
        ).model

        # Callback-based metrics (call-scoped)
        self.metrics = LLMRunMetrics()

        self.chain = (self.prompt | self.llm).with_config(callbacks=[self.metrics])

    def judge(
        self,
        *,
        question: str,
        answer: str,
        patched_scot: str,
    ) -> JudgeResult:
        # Reset metrics for THIS call only
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
                "patched_scot": patched_scot,
            }
        )

        raw = str(response.content).strip()

        label = self._extract(LABEL_PATTERN, raw, "LABEL")
        rationale = self._extract(RATIONALE_PATTERN, raw, "RATIONALE")

        return JudgeResult(
            label=label.lower(),
            rationale=rationale,
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )

    @staticmethod
    def _extract(pattern: re.Pattern, text: str, tag: str) -> str:
        match = pattern.search(text)
        if not match:
            raise ValueError(f"<{tag}> block not found in judge LLM output:\n{text}")
        return match.group(1).strip()
