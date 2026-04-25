from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined


class PromptLoader:
    """
    Loads structured LLM prompts stored as YAML files.

    Prompts are treated as immutable benchmark artifacts and
    are rendered via Jinja2 to support conditional ablation.
    """

    def __init__(self, prompt_dir: Path):
        self.prompt_dir = Path(prompt_dir).resolve()
        if not self.prompt_dir.exists():
            raise FileNotFoundError(f"Prompt directory not found: {self.prompt_dir}")

        self.env = Environment(
            undefined=StrictUndefined,  # fail fast if variable missing
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def load(self, relative_path: str, *, context: dict | None = None) -> dict:
        """
        Load and render a YAML prompt file using Jinja2.

        Args:
            relative_path: Path like "buggy_code_generator.yaml"
            context: Variables used for conditional prompting

        Returns:
            Parsed YAML as a dictionary
        """
        path = self.prompt_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            template_text = f.read()

        template = self.env.from_string(template_text)
        rendered = template.render(context or {})

        data = yaml.safe_load(rendered)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid prompt format in {path}")

        return data
