"""
Abstract base class for LLM providers.

Users can implement this interface to plug in any LLM backend
for hypothesis generation, verification, and reflection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers used by PhysMem.

    The learning pipeline (consolidation, verification, reflection) needs
    an LLM to:
    1. Generate hypotheses from experience clusters
    2. Attribute episode outcomes to hypotheses
    3. Refine principles based on prediction errors

    Implement this interface to use any LLM backend.

    Example::

        class MyLLM(BaseLLM):
            def generate(self, prompt, **kwargs):
                return my_api.call(prompt)

        mem = PhysMem(llm=MyLLM())
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt / query.
            system_prompt: Optional system-level instructions.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        pass

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
        **kwargs,
    ) -> str:
        """
        Generate text expected to be JSON.

        Default implementation just calls generate(). Override if your LLM
        supports structured output / JSON mode.
        """
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
