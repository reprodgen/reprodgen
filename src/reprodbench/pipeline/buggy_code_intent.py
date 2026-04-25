import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from reprodbench.llm.callbacks.metrics import LLMRunMetrics
from reprodbench.llm.client import LLMClient
from reprodbench.llm.prompts.prompt_loader import PromptLoader

# ============================================================
# Regex
# ============================================================

BUGGY_INTENT_PATTERN = re.compile(
    r"<BUGGY_INTENT>\s*(.*?)\s*</BUGGY_INTENT>",
    re.DOTALL | re.IGNORECASE,
)

# ============================================================
# Artifact
# ============================================================


@dataclass
class BuggyIntentArtifact:
    buggy_intent: str
    latency: Optional[float] = None
    token_usage: Optional[dict] = None


# ============================================================
# Extractor
# ============================================================


class BuggyCodeIntentExtractor:
    def __init__(
        self,
        prompt_dir,
        provider,
        model,
        temperature: float = 0.0,
    ):
        loader = PromptLoader(prompt_dir)
        prompt_def = loader.load("buggy_code_intent_generator.yaml")

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

    def extract(self, question_text: str) -> BuggyIntentArtifact:
        self.metrics.reset()

        response = self.chain.invoke({"question": question_text})

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = BUGGY_INTENT_PATTERN.search(raw)

        if not match:
            raise ValueError(f"<BUGGY_INTENT> block not found in LLM output:\n{raw}")

        artifact = BuggyIntentArtifact(
            buggy_intent=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )

        return artifact


# ============================================================
# Refiner
# ============================================================


class BuggyCodeIntentRefiner:
    def __init__(
        self,
        prompt_dir,
        provider,
        model,
        temperature: float = 0.0,
    ):
        prompt_def = PromptLoader(prompt_dir).load("buggy_code_intent_generator.yaml")

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

    # --------------------------------------------------------

    def refine(
        self,
        *,
        question: str,
        buggy_intent: str,
        judge_label: str,
        judge_rationale: str,
    ) -> BuggyIntentArtifact:
        self.metrics.reset()

        response = self.chain.invoke(
            {
                "question": question,
                "buggy_code_intent": buggy_intent,
                "label": judge_label,
                "rationale": judge_rationale,
            }
        )

        raw = str(response.content).strip().replace("\r\n", "\n")
        match = BUGGY_INTENT_PATTERN.search(raw)

        if not match:
            raise ValueError(
                f"<BUGGY_INTENT> block not found in refined output:\n{raw}"
            )

        artifact = BuggyIntentArtifact(
            buggy_intent=match.group(1).strip(),
            latency=self.metrics.latency,
            token_usage=self.metrics.token_usage,
        )

        return artifact
