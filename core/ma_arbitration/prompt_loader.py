from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

# For experts, the "full schema" prompts typically contain these markers.
_FULL_SCHEMA_MARKERS = ("applicability_label", "evidence_spans", "confidence")

def _looks_full_schema(prompt_text: str) -> bool:
    t = prompt_text.lower()
    return all(m in t for m in _FULL_SCHEMA_MARKERS)

def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path.read_text(encoding="utf-8", errors="ignore")

@dataclass(frozen=True)
class PromptLoadResult:
    path: Path
    text: str
    schema: str  # "full" or "lite"

def load_expert_prompt(
    expert_spec: Dict[str, Any],
    llm_ma_dir: Path,
    prefer_prompt_path: bool = True,
    require_full_schema: bool = True,
) -> PromptLoadResult:
    """
    Load the expert prompt for an expert entry in ma_config.json.

    Strategy:
      - Prefer 'prompt_path' (usually top3/top5 variants) if exists.
      - If require_full_schema and chosen prompt doesn't look like full schema,
        fallback to 'prompt_file' (full schema).
    """
    pf = expert_spec.get("prompt_file")
    pp = expert_spec.get("prompt_path")
    if not pf and not pp:
        raise ValueError(f"expert spec has no prompt paths: {expert_spec}")

    cand_paths = []
    if prefer_prompt_path and pp:
        cand_paths.append(llm_ma_dir / pp)
    if pf:
        cand_paths.append(llm_ma_dir / pf)
    if not prefer_prompt_path and pp:
        cand_paths.append(llm_ma_dir / pp)

    chosen = None
    chosen_text = None
    for p in cand_paths:
        if p.exists():
            chosen = p
            chosen_text = read_text(p)
            break
    if chosen is None or chosen_text is None:
        raise FileNotFoundError(
            f"Cannot find expert prompt for {expert_spec.get('agent_id')}: tried {cand_paths}"
        )

    schema = "full" if _looks_full_schema(chosen_text) else "lite"

    # Fallback to full schema prompt if needed
    if require_full_schema and schema != "full" and pf:
        fb = llm_ma_dir / pf
        if fb.exists():
            fb_text = read_text(fb)
            if _looks_full_schema(fb_text):
                return PromptLoadResult(path=fb, text=fb_text, schema="full")

    return PromptLoadResult(path=chosen, text=chosen_text, schema=schema)

def load_arbiter_prompt(arbiter_prompt_file: str, llm_ma_dir: Path) -> PromptLoadResult:
    p = llm_ma_dir / arbiter_prompt_file
    txt = read_text(p)
    # Arbiter prompt schema markers differ; we still report "full/lite" using the same heuristic.
    schema = "full" if _looks_full_schema(txt) else "lite"
    return PromptLoadResult(path=p, text=txt, schema=schema)

def load_all_prompts(cfg: Dict[str, Any], llm_ma_dir: Path) -> Tuple[Dict[str, PromptLoadResult], PromptLoadResult]:
    experts = cfg.get("experts") or []
    expert_prompts: Dict[str, PromptLoadResult] = {}
    for e in experts:
        agent_id = e.get("agent_id", "unknown")
        expert_prompts[agent_id] = load_expert_prompt(
            e,
            llm_ma_dir=llm_ma_dir,
            prefer_prompt_path=True,
            require_full_schema=True,
        )

    arbiter_file = cfg.get("arbiter_prompt_file")
    if not arbiter_file:
        raise ValueError("ma_config.json missing 'arbiter_prompt_file'")
    arbiter_prompt = load_arbiter_prompt(arbiter_file, llm_ma_dir=llm_ma_dir)
    return expert_prompts, arbiter_prompt
