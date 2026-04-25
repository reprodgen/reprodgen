import time
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler


class LLMRunMetrics(BaseCallbackHandler):
    """
    Collects latency and token usage in a provider-agnostic way.
    Works for OpenAI, Ollama, Anthropic, local models.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.latency: Optional[float] = None

        self.token_usage: Optional[Dict[str, Any]] = None
        self.model_name: Optional[str] = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()

    def on_llm_end(self, response, **kwargs):
        self.end_time = time.time()
        self.latency = self.end_time - self.start_time

        # 1️⃣ Preferred: standardized LangChain metadata
        try:
            gen = response.generations[0][0]
            msg = gen.message

            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                self.token_usage = msg.usage_metadata

            # model name (best effort)
            self.model_name = getattr(msg, "response_metadata", {}).get("model_name")
        except Exception:
            pass

        # Fallback: provider-specific llm_output
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            if self.token_usage is None:
                if "token_usage" in llm_output:  # OpenAI
                    self.token_usage = llm_output["token_usage"]
                elif "usage" in llm_output:  # Anthropic
                    self.token_usage = llm_output["usage"]

            if self.model_name is None:
                self.model_name = llm_output.get("model_name")

    def on_llm_error(self, error, **kwargs):
        self.end_time = time.time()
        if self.start_time:
            self.latency = self.end_time - self.start_time
