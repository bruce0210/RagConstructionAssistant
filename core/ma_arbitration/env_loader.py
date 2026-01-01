from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

_EXPORT_RE = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$')

def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1]
    v = v.replace(r'\"', '"').replace(r"\'", "'")
    return v

def parse_env_sh(path: Path) -> Dict[str, str]:
    """
    Parse a simple env.sh that contains lines like:
      export OPENAI_API_KEY="..."
      export DASHSCOPE_API_KEY="..."
    Returns {VAR: value}. Ignores comments/empty lines.
    """
    if not path.exists():
        raise FileNotFoundError(f"env.sh not found: {path}")
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _EXPORT_RE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        out[k] = _strip_quotes(v)
    return out
