from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class LLMClient:
    """
    Thin OOP wrapper around LangChain chat models.
    Responsible for model initialization and invocation.
    """

    def __init__(
        self,
        model_name: str,
        provider: Literal["openai", "ollama"],
        temperature: float = 1.0,
    ):
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.model = self._init_llm()

    def _init_llm(self):
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
            )

        if self.provider == "ollama":
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
            )

        if self.provider == "anthropic":
            return ChatAnthropic(
                model_name=self.model_name,  # satisfies stub
                temperature=self.temperature,
                timeout=None,  # satisfies stub
                stop=None,  # satisfies stub
            )

        raise ValueError(f"Unsupported provider: {self.provider}")
