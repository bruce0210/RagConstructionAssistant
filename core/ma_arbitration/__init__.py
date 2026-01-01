"""
Multi-agent arbitration (MA) utilities.

Used by Streamlit UI:
- 6 discipline experts: Qwen via DashScope (OpenAI-compatible)
- 1 arbiter: OpenAI

Consistent with the original llm_ma experiments.
"""

from .engine import (
    default_llm_ma_dir,
    load_ma_config,
    load_ma_resources,
    validate_ma_install,
    run_ma_for_ui,
)

__all__ = [
    "default_llm_ma_dir",
    "load_ma_config",
    "load_ma_resources",
    "validate_ma_install",
    "run_ma_for_ui",
]
