"""
Provider-agnostic LLM client for the Auto Overseer.

Supported providers:
  anthropic    — Anthropic SDK (default); env: ANTHROPIC_API_KEY
  openai       — OpenAI SDK; env: OPENAI_API_KEY
  deepseek     — OpenAI-compatible; env: DEEPSEEK_API_KEY, base_url auto-set
  openai_compatible — any OpenAI-compatible endpoint; set LLM_BASE_URL

Configure via env vars (or pass kwargs to LLMClient):
  LLM_PROVIDER   = anthropic | openai | deepseek | openai_compatible
  LLM_MODEL      = model name (e.g. claude-haiku-4-5-20251001, gpt-4o-mini)
  LLM_API_KEY    = override key for any provider
  LLM_BASE_URL   = base URL for openai_compatible endpoints (e.g. http://localhost:11434/v1)
"""
import os

PROVIDER_DEFAULTS = {
    "anthropic":         {"model": "claude-haiku-4-5-20251001",  "base_url": None},
    "openai":            {"model": "gpt-4o-mini",                "base_url": None},
    "deepseek":          {"model": "deepseek-chat",              "base_url": "https://api.deepseek.com/v1"},
    "openai_compatible": {"model": "llama3",                     "base_url": "http://localhost:11434/v1"},
}

PROVIDER_KEY_ENVS = {
    "anthropic":         ["ANTHROPIC_API_KEY"],
    "openai":            ["OPENAI_API_KEY"],
    "deepseek":          ["DEEPSEEK_API_KEY", "DEEP_SEEK_API_KEY"],  # support both spellings
    "openai_compatible": ["LLM_API_KEY"],
}


class LLMClient:
    """Thin wrapper that normalises Anthropic and OpenAI-compatible APIs."""

    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 max_tokens: int = 4096):
        self.provider   = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
        defaults        = PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS["anthropic"])
        self.model      = model or os.environ.get("LLM_MODEL", defaults["model"])
        self.max_tokens = max_tokens
        key_envs = PROVIDER_KEY_ENVS.get(self.provider, ["LLM_API_KEY"])
        found_key = next((os.environ.get(k) for k in key_envs if os.environ.get(k)), "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or found_key
        self.base_url   = (base_url
                           or os.environ.get("LLM_BASE_URL")
                           or defaults.get("base_url"))
        self._client    = self._build_client()

    def _build_client(self):
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key or None)
        else:
            import openai
            kwargs = {"api_key": self.api_key or "placeholder"}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return openai.OpenAI(**kwargs)

    def chat(self, system: str, user: str) -> str:
        """Send a chat request and return the raw response text. Returns '' on error."""
        try:
            if self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            else:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                return resp.choices[0].message.content
        except Exception as exc:
            print(f"  [LLMClient] {self.provider}/{self.model} error: {exc}")
            return ""

    def __repr__(self):
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"
