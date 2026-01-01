from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import time

try:
    import requests
except Exception as e:
    raise ImportError("Missing dependency: requests. Please install it (pip install requests).") from e


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str
    timeout: int = 180


def _post_chat_completions(
    cfg: ProviderConfig,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> str:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key.strip()}",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
    # raise_for_status will include status code but not our secrets
    r.raise_for_status()
    data = r.json()

    # OpenAI-compatible response
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # Provide a helpful error without leaking anything sensitive
        raise RuntimeError(f"Unexpected completion response schema: keys={list(data.keys())}")


def chat_with_retry(
    cfg: ProviderConfig,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1200,
    retries: int = 2,
    sleep_sec: float = 0.5,
) -> str:
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            return _post_chat_completions(
                cfg=cfg,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(sleep_sec * (i + 1))
    assert last_err is not None
    raise last_err


def default_dashscope_base_url() -> str:
    # DashScope OpenAI-compatible endpoint
    return os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def default_openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
