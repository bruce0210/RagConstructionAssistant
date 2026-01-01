from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from .env_loader import parse_env_sh
from .prompt_loader import load_all_prompts, PromptLoadResult
from .providers import (
    ProviderConfig,
    chat_with_retry,
    default_dashscope_base_url,
    default_openai_base_url,
)

# --------------------------
# Resource loading (Step-1)
# --------------------------

@dataclass
class MAResources:
    llm_ma_dir: Path
    cfg: Dict[str, Any]
    env: Dict[str, str]
    expert_prompts: Dict[str, PromptLoadResult]
    arbiter_prompt: PromptLoadResult

def repo_root_from_this_file() -> Path:
    # .../core/ma_arbitration/engine.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]

def default_llm_ma_dir(repo_root: Optional[Path] = None) -> Path:
    rr = repo_root or repo_root_from_this_file()
    return rr / "experiments" / "retrieval" / "data" / "llm_ma"

def load_ma_config(llm_ma_dir: Path) -> Dict[str, Any]:
    cfg_path = llm_ma_dir / "configs" / "ma_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"ma_config.json not found: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore"))

def load_ma_resources(llm_ma_dir: Optional[Path] = None) -> MAResources:
    llm_ma_dir = llm_ma_dir or default_llm_ma_dir()
    cfg = load_ma_config(llm_ma_dir)
    env = parse_env_sh(llm_ma_dir / "configs" / "env.sh")
    expert_prompts, arbiter_prompt = load_all_prompts(cfg, llm_ma_dir)
    return MAResources(
        llm_ma_dir=llm_ma_dir,
        cfg=cfg,
        env=env,
        expert_prompts=expert_prompts,
        arbiter_prompt=arbiter_prompt,
    )

def validate_ma_install(llm_ma_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Smoke check (NO model calls):
      - config exists and parses
      - env.sh exists and contains expected keys (only reports presence, not values)
      - prompts exist and load
    """
    llm_ma_dir = llm_ma_dir or default_llm_ma_dir()
    report: Dict[str, Any] = {"ok": False, "llm_ma_dir": str(llm_ma_dir), "errors": []}
    try:
        res = load_ma_resources(llm_ma_dir)
        report["env"] = {
            "OPENAI_API_KEY": bool(res.env.get("OPENAI_API_KEY")),
            "DASHSCOPE_API_KEY": bool(res.env.get("DASHSCOPE_API_KEY")),
        }
        report["cfg_summary"] = {
            "schema_version": res.cfg.get("schema_version"),
            "expert_model_primary": res.cfg.get("expert_model_primary"),
            "arbiter_model": res.cfg.get("arbiter_model"),
            "candidates_top_k": res.cfg.get("candidates_top_k"),
            "experts": [e.get("agent_id") for e in (res.cfg.get("experts") or [])],
            "arbiter_prompt_file": res.cfg.get("arbiter_prompt_file"),
        }
        report["prompt_paths"] = {
            "experts": {k: str(v.path) for k, v in res.expert_prompts.items()},
            "arbiter": str(res.arbiter_prompt.path),
        }
        report["prompt_schema"] = {
            "experts": {k: v.schema for k, v in res.expert_prompts.items()},
            "arbiter": res.arbiter_prompt.schema,
        }
        report["routing_check"] = {
            "experts_should_be_qwen": str(res.cfg.get("expert_model_primary", "")).lower().startswith("qwen"),
            "arbiter_should_be_openai": str(res.cfg.get("arbiter_model", "")).lower().startswith("gpt"),
        }
        report["ok"] = True
        return report
    except Exception as e:
        report["errors"].append(repr(e))
        return report

# --------------------------
# MA runner (Step-2)
# --------------------------

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

def extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from a model response.
    - Accepts raw JSON, or fenced JSON, or extra text surrounding JSON.
    - Repairs a common trailing comma issue.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output; cannot parse JSON.")

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # remove fences
    t2 = _JSON_FENCE_RE.sub("", text).strip()

    # locate { ... }
    i = t2.find("{")
    j = t2.rfind("}")
    if i >= 0 and j > i:
        cand = t2[i:j + 1]
        try:
            return json.loads(cand)
        except Exception:
            # mild cleanup
            cand2 = re.sub(r",\s*}", "}", cand)
            cand2 = re.sub(r",\s*]", "]", cand2)
            return json.loads(cand2)

    raise ValueError("Cannot extract JSON object from output.")

def normalize_candidates(raw: Any, top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Normalize candidate formats to:
      [{"clause_id": str, "text": str, "meta": dict}, ...]
    """
    if raw is None:
        return []
    out: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        xs = raw[:top_k]
        for x in xs:
            if isinstance(x, dict):
                cid = x.get("clause_id") or x.get("id") or x.get("cid") or x.get("clauseId")
                txt = x.get("text") or x.get("content") or x.get("chunk") or x.get("passage") or ""
                meta = {k: v for k, v in x.items() if k not in ("text", "content", "chunk", "passage")}
                out.append({"clause_id": str(cid) if cid is not None else None, "text": str(txt), "meta": meta})
            else:
                out.append({"clause_id": None, "text": str(x), "meta": {}})
        return out

    if isinstance(raw, dict):
        # allow {"candidates":[...]} etc.
        for key in ("candidates", "hits", "results", "data"):
            if key in raw and isinstance(raw[key], list):
                return normalize_candidates(raw[key], top_k=top_k)
        # fallback: treat whole dict as one candidate
        cid = raw.get("clause_id") or raw.get("id")
        txt = raw.get("text") or raw.get("content") or ""
        meta = {k: v for k, v in raw.items() if k not in ("text", "content")}
        return [{"clause_id": str(cid) if cid is not None else None, "text": str(txt), "meta": meta}]

    # fallback: stringify
    return [{"clause_id": None, "text": str(raw), "meta": {}}]

def build_candidates_block(cands: List[Dict[str, Any]], max_chars_each: int = 900) -> str:
    """
    Build a readable candidates block for prompts.
    """
    blocks: List[str] = []
    for idx, c in enumerate(cands, start=1):
        cid = c.get("clause_id") or ""
        txt = (c.get("text") or "").strip().replace("\r", " ")
        if len(txt) > max_chars_each:
            txt = txt[:max_chars_each] + "…"
        blocks.append(f"{idx}. [clause_id={cid}]\n{txt}\n")
    return "\n".join(blocks).strip()

def disagreement_and_trigger(expert_objs: List[Dict[str, Any]], trig_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine whether to trigger arbiter based on expert disagreement.
    """
    top_clauses: List[str] = []
    appls: List[str] = []
    evidence_missing = False

    for e in expert_objs:
        sel = e.get("selected_clause_ids") or []
        if sel:
            top_clauses.append(str(sel[0]))
        a = e.get("applicability_label")
        if a is not None:
            appls.append(str(a))
        ev = e.get("evidence_spans") or []
        # if expert claims relevant but provides no evidence, treat as missing
        if (e.get("is_relevant_to_discipline") in ("yes", "partial")) and not ev:
            evidence_missing = True

    clause_disagree = len(set(top_clauses)) >= 2
    applicability_disagree = len(set(appls)) >= 2

    trigger = False
    reasons: List[str] = []
    if trig_cfg.get("clause_disagree", True) and clause_disagree:
        trigger = True
        reasons.append("clause_disagree")
    if trig_cfg.get("applicability_disagree", True) and applicability_disagree:
        trigger = True
        reasons.append("applicability_disagree")
    if trig_cfg.get("evidence_missing", True) and evidence_missing:
        trigger = True
        reasons.append("evidence_missing")

    diag = {
        "top_clauses": top_clauses,
        "applicabilities": appls,
        "clause_disagree": clause_disagree,
        "applicability_disagree": applicability_disagree,
        "evidence_missing": evidence_missing,
        "trigger_reasons": reasons,
    }
    return trigger, diag

def aggregate_without_arbiter(expert_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fallback aggregation when arbiter is not triggered.
    Produces the same schema as the arbiter output (best effort).
    """
    votes: List[str] = []
    for e in expert_objs:
        sel = e.get("selected_clause_ids") or []
        if sel:
            votes.append(str(sel[0]))

    final_clause_ids: List[str] = []
    if votes:
        cnt = Counter(votes)
        best = max(cnt.values())
        tied = [c for c, n in cnt.items() if n == best]
        # deterministic: keep 1 if clear, otherwise keep all tied
        final_clause_ids = [tied[0]] if len(tied) == 1 else tied

    # majority applicability among experts who voted for the first chosen clause
    chosen = final_clause_ids[0] if final_clause_ids else None
    appl_votes: List[str] = []
    conds: List[str] = []
    citations: List[Dict[str, str]] = []

    for e in expert_objs:
        sel = e.get("selected_clause_ids") or []
        if chosen and sel and str(sel[0]) == chosen:
            a = e.get("applicability_label")
            if a is not None:
                appl_votes.append(str(a))
            for c in (e.get("conditions") or []):
                if isinstance(c, str) and c.strip():
                    conds.append(c.strip())
            for ev in (e.get("evidence_spans") or []):
                if isinstance(ev, dict):
                    cid2 = ev.get("clause_id")
                    sp = (ev.get("span") or "").strip()
                    if cid2 and sp:
                        citations.append({"clause_id": str(cid2), "span": sp})

    final_app = "unknown"
    if appl_votes:
        cnt2 = Counter(appl_votes)
        final_app = cnt2.most_common(1)[0][0]

    # dedup conditions (keep order)
    seen = set()
    final_conditions: List[str] = []
    for c in conds:
        if c not in seen:
            seen.add(c)
            final_conditions.append(c)

    # Construct a conservative final answer
    if chosen:
        answer = (
            f"未触发仲裁：基于多数专家一致意见，最相关条款为 clause_id={chosen}。"
            f"适用性判断为 {final_app}。"
        )
        if final_conditions:
            answer += " 需满足条件/注意事项：" + "；".join(final_conditions[:6])
    else:
        answer = "未触发仲裁：专家未能从候选池中选出明确相关条款，建议扩大候选范围或调整问题表述。"

    return {
        "final_clause_ids": final_clause_ids,
        "final_applicability_label": final_app,
        "final_conditions": final_conditions,
        "final_answer": answer,
        "citations": citations[:20],
        "accepted_experts": [e.get("agent_id") for e in expert_objs if e.get("agent_id")],
        "rejected_points": [],
        "decision_notes": "No arbiter invoked; majority-based aggregation (best effort).",
    }

def build_evidence_bundle(
    query: str,
    candidates: List[Dict[str, Any]],
    experts: List[Dict[str, Any]],
    trigger_diag: Dict[str, Any],
    final: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Bundle for DB evidence (JSON).
    Important: avoid storing full candidate text; keep only a short preview.
    Experts/arbiter JSON are stored in full (as required).
    """
    cand_slim = []
    for c in candidates:
        txt = (c.get("text") or "").strip().replace("\r", " ")
        cand_slim.append({
            "clause_id": c.get("clause_id"),
            "meta": c.get("meta") or {},
            "text_preview": (txt[:180] + "…") if len(txt) > 180 else txt,
        })

    return {
        "mode": "ma",
        "query": query,
        "cfg_summary": {
            "schema_version": cfg.get("schema_version"),
            "expert_model_primary": cfg.get("expert_model_primary"),
            "expert_model_fallbacks": cfg.get("expert_model_fallbacks"),
            "arbiter_model": cfg.get("arbiter_model"),
            "candidates_top_k": cfg.get("candidates_top_k"),
        },
        "candidates": cand_slim,
        "experts": experts,          # full JSON (per your requirement)
        "trigger_diag": trigger_diag,
        "final": final,              # full arbiter/final JSON
    }

def run_ma_for_ui(
    query: str,
    candidates_raw: Any,
    llm_ma_dir: Optional[Path] = None,
    candidates_topn: int = 50,
    temperature: float = 0.0,
    max_tokens_expert: int = 1200,
    max_tokens_arbiter: int = 1400,
) -> Dict[str, Any]:
    """
    Main entry for UI:
      - experts (6) use DashScope/Qwen (OpenAI-compatible endpoint)
      - arbiter uses OpenAI
    Returns dict with experts list, trigger flag/diag, final decision, and evidence_bundle.
    """
    res = load_ma_resources(llm_ma_dir)
    cfg = res.cfg

    dash_key = res.env.get("DASHSCOPE_API_KEY") or ""
    openai_key = res.env.get("OPENAI_API_KEY") or ""
    if not dash_key:
        raise RuntimeError("DASHSCOPE_API_KEY missing in llm_ma/configs/env.sh")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY missing in llm_ma/configs/env.sh")

    dash_cfg = ProviderConfig(base_url=default_dashscope_base_url(), api_key=dash_key, timeout=180)
    openai_cfg = ProviderConfig(base_url=default_openai_base_url(), api_key=openai_key, timeout=180)

    expert_model_primary = str(cfg.get("expert_model_primary") or "").strip()
    expert_model_fallbacks = cfg.get("expert_model_fallbacks") or []
    arbiter_model = str(cfg.get("arbiter_model") or "").strip()

    # enforce your experiment design: experts must be qwen*, arbiter must be gpt*
    models_to_try = [expert_model_primary] + [m for m in expert_model_fallbacks if isinstance(m, str)]
    models_to_try = [m for m in models_to_try if m and m.lower().startswith("qwen")]

    if not models_to_try:
        raise RuntimeError(f"No valid Qwen expert models found in config: expert_model_primary={expert_model_primary}, fallbacks={expert_model_fallbacks}")
    if not arbiter_model.lower().startswith("gpt"):
        # allow but warn via returned meta (do not print)
        pass

    top_k_cfg = int(cfg.get("candidates_top_k") or 30)
    trig_cfg = cfg.get("arbitration_trigger") or {}

    # normalize candidates (UI will pass ~Top50)
    cands_all = normalize_candidates(candidates_raw, top_k=candidates_topn)
    # use config top_k for prompt
    cands = cands_all[:max(1, min(len(cands_all), top_k_cfg))]
    cand_block = build_candidates_block(cands, max_chars_each=900)

    expert_outputs: List[Dict[str, Any]] = []
    used_models: Dict[str, str] = {}

    # 6 experts
    for e in (cfg.get("experts") or []):
        aid = e.get("agent_id")
        if not aid:
            continue

        sys_prompt = res.expert_prompts[aid].text
        expert_topn = int(e.get("top_n") or 3)

        user_msg = (
            f"【用户问题】\n{query}\n\n"
            f"【候选条款（Top-{len(cands)}）】\n{cand_block}\n\n"
            f"请严格按你的输出JSON schema返回。\n"
            f"重要：若候选条款都不相关：selected_clause_ids 置为空数组 []。\n"
            f"若相关，请按相关性从高到低输出 Top-{expert_topn}（最多 {expert_topn} 条）。"
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        content = None
        last_err = None
        chosen_model = None

        for m in models_to_try:
            try:
                content = chat_with_retry(
                    cfg=dash_cfg,
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens_expert,
                    retries=2,
                    sleep_sec=0.6,
                )
                chosen_model = m
                break
            except Exception as ex:
                last_err = ex
                continue

        if content is None:
            raise RuntimeError(f"Expert call failed for agent={aid}: {repr(last_err)}")

        obj = extract_json_obj(content)
        # enforce agent_id if missing
        if "agent_id" not in obj:
            obj["agent_id"] = aid
        expert_outputs.append(obj)
        if chosen_model:
            used_models[aid] = chosen_model

    trigger, diag = disagreement_and_trigger(expert_outputs, trig_cfg)

    if trigger:
        experts_json = json.dumps(expert_outputs, ensure_ascii=False)
        arb_user = (
            f"【用户问题】\n{query}\n\n"
            f"【候选条款（Top-{len(cands)}）】\n{cand_block}\n\n"
            f"【6位专家意见JSON数组】\n{experts_json}\n\n"
            f"请严格按你输出JSON schema返回裁决。"
        )
        messages = [
            {"role": "system", "content": res.arbiter_prompt.text},
            {"role": "user", "content": arb_user},
        ]
        arb_content = chat_with_retry(
            cfg=openai_cfg,
            model=arbiter_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens_arbiter,
            retries=2,
            sleep_sec=0.8,
        )
        final = extract_json_obj(arb_content)
    else:
        final = aggregate_without_arbiter(expert_outputs)

    evidence_bundle = build_evidence_bundle(
        query=query,
        candidates=cands,
        experts=expert_outputs,
        trigger_diag=diag,
        final=final,
        cfg=cfg,
    )

    return {
        "query": query,
        "candidates_used": cands,
        "experts": expert_outputs,
        "triggered": bool(trigger),
        "trigger_diag": diag,
        "final": final,
        "used_models": used_models,
        "evidence_bundle": evidence_bundle,
    }
