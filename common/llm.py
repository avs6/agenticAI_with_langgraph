import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load .env once (safe to call multiple times)
load_dotenv()

@lru_cache(maxsize=4)
def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
):
    """
    Global LLM factory with caching.
    Switch providers via env or per-call overrides.

    Env:
      LLM_PROVIDER=openai|ollama|google_genai
      LLM_MODEL=...
      OPENAI_API_KEY=...
      OPENAI_BASE_URL=... (optional)
      OLLAMA_BASE_URL=http://localhost:11434
      GOOGLE_API_KEY=...
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip()
    model = (model or os.getenv("LLM_MODEL", "llama3.1")).strip()

    # Normalize provider name
    if provider in ("google", "gemini"):
        provider = "google_genai"

    model_id = f"{provider}:{model}"

    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature

    # Provider-specific wiring
    if provider == "openai":
        # For OpenAI-compatible gateways (vLLM, etc.), set OPENAI_BASE_URL
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

    elif provider == "ollama":
        kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    elif provider == "google_genai":
        # Usually picked up automatically, but safe to pass
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    return init_chat_model(model_id, **kwargs)


def reset_llm_cache():
    """Call this if you change env vars at runtime and want a fresh client."""
    get_llm.cache_clear()
