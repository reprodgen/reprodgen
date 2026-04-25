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
# Result artifact
# ============================================================


@dataclass
class JudgeResult:
    label: str
    rationale: str
    latency: float | None = None
    token_usage: dict | None = None


# ============================================================
# Patched code judge
# ============================================================


class PatchedCodeJudge:
    """
    Evaluates whether a patched Python script:

      - Executes successfully
      - Faithfully implements the fix described
        in the accepted Stack Overflow answer
      - Does NOT reintroduce the buggy behavior
      - Uses the intended API / approach from the answer

    This judge is stateless and must rely ONLY on:
      - question
      - accepted answer
      - patched code
      - execution behavior (stdout / stderr / exit_code)
    """

    def __init__(
        self,
        prompt_dir: Path,
        provider: str,
        model: str,
        temperature: float = 0.0,
    ):
        loader = PromptLoader(prompt_dir)
        prompt_def = loader.load("judge_patched_code_semantic.yaml")

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

    # --------------------------------------------------------
    # Judge entry point
    # --------------------------------------------------------

    def judge(
        self,
        *,
        question: str,
        answer: str,
        patched_code: str,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> JudgeResult:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
                "patched_code": patched_code,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }
        )

        raw = str(response.content).strip()
        print("Patched Judge LLM output:\n", raw)

        label = self._extract(LABEL_PATTERN, raw, "LABEL")
        rationale = self._extract(RATIONALE_PATTERN, raw, "RATIONALE")

        return JudgeResult(
            label=label.lower(),
            rationale=rationale,
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _extract(pattern: re.Pattern, text: str, tag: str) -> str:
        match = pattern.search(text)
        if not match:
            raise ValueError(
                f"<{tag}> block not found in patched judge LLM output:\n{text}"
            )
        return match.group(1).strip()
