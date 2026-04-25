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


class BuggyScotJudge:
    """
    Judges whether a generated buggy SCoT (Structured Chain of Thought)
    faithfully reflects the user's procedural reasoning and control flow
    that leads to the buggy behavior described in the Stack Overflow question.
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        loader = PromptLoader(prompt_dir)
        prompt_def = loader.load("judge_buggy_scot.yaml")

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

    def judge(
        self,
        *,
        question: str,
        buggy_scot: str,
    ) -> JudgeResult:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "buggy_scot": buggy_scot,
            }
        )

        raw = str(response.content).strip()
        # print("Buggy SCoT Judge LLM output:\n", raw)

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
