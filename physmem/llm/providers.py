"""
Built-in LLM provider implementations.

Supports OpenAI, Google Gemini, Alibaba Qwen, and Kimi (Moonshot).
All providers are optional - install only what you need.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from physmem.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI GPT models (GPT-4o, GPT-5.1, o3, etc.)

    Requires: pip install openai>=1.0.0
    API Key: OPENAI_API_KEY or OPENROUTER_API_KEY env var.

    Args:
        model: Model name (default: "gpt-4o")
        api_key: API key (or set env var)
        base_url: Custom base URL (e.g., for OpenRouter)
        use_openrouter: Use OpenRouter endpoint (default: False)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_openrouter: bool = False,
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai>=1.0.0")

        self.model = model

        if use_openrouter or (base_url and "openrouter" in base_url.lower()):
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("Set OPENROUTER_API_KEY env var or pass api_key.")
            self.base_url = base_url or "https://openrouter.ai/api/v1"
            if not self.model.startswith("openai/"):
                self.model = f"openai/{self.model}"
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Set OPENAI_API_KEY env var or pass api_key.")
            self.base_url = base_url

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"[OpenAI] API error: {e}")
            return ""


class GeminiLLM(BaseLLM):
    """
    Google Gemini models.

    Requires: pip install google-genai
    API Key: GOOGLE_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Set GOOGLE_API_KEY env var or pass api_key.")
        self.model = model
        self.enable_thinking = enable_thinking

        try:
            from google import genai
            from google.genai import types
            self._types = types
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("google-genai package required. Install with: pip install google-genai")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        contents = []
        if system_prompt:
            contents.append(system_prompt + "\n\n")
        contents.append(prompt)

        try:
            config = self._types.GenerateContentConfig(temperature=temperature)
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            try:
                return response.candidates[0].content.parts[0].text.strip()
            except (IndexError, AttributeError):
                return ""
        except Exception as e:
            print(f"[Gemini] API error: {e}")
            return ""


class QwenLLM(BaseLLM):
    """
    Alibaba Qwen models via DashScope API.

    Requires: pip install openai>=1.0.0
    API Key: DASHSCOPE_API_KEY or QWEN_API_KEY env var.
    """

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model: str = "qwen-plus-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs,
    ):
        self.api_key = (
            api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        )
        if not self.api_key:
            raise ValueError("Set DASHSCOPE_API_KEY or QWEN_API_KEY env var.")
        self.model = model
        self.base_url = base_url or self.DASHSCOPE_BASE_URL
        self.enable_thinking = enable_thinking

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai>=1.0.0")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if self.enable_thinking:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"enable_thinking": True},
                    stream=True,
                )
                parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                return "".join(parts).strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Qwen] API error: {e}")
            return ""


class KimiLLM(OpenAILLM):
    """
    Kimi (Moonshot AI) models.

    API Key: MOONSHOT_API_KEY or KIMI_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "kimi-k2-0905-preview",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        api_key = api_key or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
        if not api_key:
            raise ValueError("Set MOONSHOT_API_KEY or KIMI_API_KEY env var.")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.moonshot.ai/v1",
            **kwargs,
        )


# ============================================================================
# Factory
# ============================================================================

PROVIDERS = {
    "openai": OpenAILLM,
    "gpt": OpenAILLM,
    "gemini": GeminiLLM,
    "google": GeminiLLM,
    "qwen": QwenLLM,
    "alibaba": QwenLLM,
    "dashscope": QwenLLM,
    "kimi": KimiLLM,
    "moonshot": KimiLLM,
}

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gpt": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "google": "gemini-2.5-flash",
    "qwen": "qwen-plus-latest",
    "alibaba": "qwen-plus-latest",
    "dashscope": "qwen-plus-latest",
    "kimi": "kimi-k2-0905-preview",
    "moonshot": "kimi-k2-0905-preview",
}


def create_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLM:
    """
    Create an LLM instance.

    Args:
        provider: "openai", "gemini", "qwen", or "kimi"
        model: Model name (uses default if not specified)
        **kwargs: Additional arguments (api_key, base_url, etc.)

    Returns:
        BaseLLM instance
    """
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(PROVIDERS.keys())}")
    model = model or DEFAULT_MODELS[provider]
    return PROVIDERS[provider](model=model, **kwargs)
