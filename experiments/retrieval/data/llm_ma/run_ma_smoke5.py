import os
import re
import json
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import requests

DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def log_line(path: str, msg: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from model output robustly.
    """
    text = (text or "").strip()
    # fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # strip code fences
    text2 = re.sub(r"^```(?:json)?\s*", "", text)
    text2 = re.sub(r"\s*```$", "", text2).strip()

    i = text2.find("{")
    j = text2.rfind("}")
    if i >= 0 and j >= 0 and j > i:
        cand = text2[i:j + 1]
        try:
            return json.loads(cand)
        except Exception:
            # mild cleanup
            cand2 = re.sub(r",\s*}", "}", cand)
            cand2 = re.sub(r",\s*]", "]", cand2)
            return json.loads(cand2)

    raise ValueError("Cannot extract JSON object from output")

def chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1200,
    timeout: int = 180,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def pick_field(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None

def normalize_candidates(raw: Any, top_k: int = 30) -> List[Dict[str, Any]]:
    """
    Normalize various candidate formats to:
    [{"clause_id":..., "text":..., "meta":...}, ...]
    """
    if raw is None:
        return []

    if isinstance(raw, list):
        out: List[Dict[str, Any]] = []
        for x in raw[:top_k]:
            if isinstance(x, dict):
                cid = pick_field(x, ["clause_id", "clauseId", "id", "cid"])
                txt = pick_field(x, ["text", "clause_text", "content", "chunk", "passage"])
                meta = {k: v for k, v in x.items() if k not in ["text", "clause_text", "content", "chunk", "passage"]}
                out.append({"clause_id": cid, "text": txt, "meta": meta})
            else:
                out.append({"clause_id": None, "text": str(x), "meta": {}})
        return out

    if isinstance(raw, dict):
        for key in ["candidates", "cand", "topk", "hits", "passages", "items"]:
            if key in raw and isinstance(raw[key], list):
                return normalize_candidates(raw[key], top_k=top_k)

    return []

def indent_lines(s: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in (s or "").split("\n"))

def build_candidates_block(cands: List[Dict[str, Any]], max_chars_each: int = 1200) -> str:
    lines: List[str] = []
    for i, c in enumerate(cands, 1):
        cid = c.get("clause_id")
        txt = (c.get("text") or "").strip().replace("\r\n", "\n")
        if len(txt) > max_chars_each:
            txt = txt[:max_chars_each] + " ...[TRUNCATED]"
        lines.append(f"[{i}] clause_id: {cid}\n{indent_lines(txt, '    ')}")
    return "\n\n".join(lines)

def disagreement_and_trigger(expert_objs: List[Dict[str, Any]], trig_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    top_clauses: List[str] = []
    appls: List[str] = []
    evidence_missing = False

    for e in expert_objs:
        sel = e.get("selected_clause_ids") or []
        if sel:
            top_clauses.append(sel[0])
            a = e.get("applicability_label")
            if a is not None:
                appls.append(str(a))
            ev = e.get("evidence_spans") or []
            if not ev:
                evidence_missing = True

    clause_disagree = (len(set(top_clauses)) > 1)
    applicability_disagree = (len(set(appls)) > 1) if appls else False

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
    votes: List[str] = []
    conf_by_clause: Dict[str, float] = {}
    appl_votes: List[str] = []
    conditions: List[str] = []
    citations: List[Dict[str, str]] = []

    for e in expert_objs:
        sel = e.get("selected_clause_ids") or []
        conf = float(e.get("confidence") or 0.0)
        if sel:
            cid = sel[0]
            votes.append(cid)
            conf_by_clause[cid] = max(conf_by_clause.get(cid, 0.0), conf)

        a = e.get("applicability_label")
        if a:
            appl_votes.append(str(a))

        for c in (e.get("conditions") or []):
            c2 = str(c).strip()
            if c2:
                conditions.append(c2)

        for ev in (e.get("evidence_spans") or []):
            if isinstance(ev, dict):
                cid2 = ev.get("clause_id")
                sp = (ev.get("span") or "").strip()
                if cid2 and sp:
                    citations.append({"clause_id": cid2, "span": sp})

    # majority vote on clause
    final_clause_ids: List[str] = []
    if votes:
        counts = Counter(votes)
        best_count = max(counts.values())
        tied = [c for c, n in counts.items() if n == best_count]
        if len(tied) == 1:
            final_clause_ids = [tied[0]]
        else:
            final_clause_ids = [sorted(tied, key=lambda c: conf_by_clause.get(c, 0.0), reverse=True)[0]]

    # majority on applicability
    final_appl = "unknown"
    if appl_votes:
        final_appl = Counter(appl_votes).most_common(1)[0][0]

    # dedup
    cond_uniq: List[str] = []
    seen = set()
    for c in conditions:
        if c not in seen:
            seen.add(c)
            cond_uniq.append(c)

    cites_uniq: List[Dict[str, str]] = []
    seen2 = set()
    for ev in citations:
        key = (ev["clause_id"], ev["span"])
        if key not in seen2:
            seen2.add(key)
            cites_uniq.append(ev)

    if final_clause_ids:
        final_answer = f"建议优先参考候选条款：{final_clause_ids[0]}。适用性：{final_appl}。如需更精确结论，请补充项目条件或扩大候选条款范围。"
    else:
        final_answer = "当前候选池中未发现足以支持结论的条款（可能缺失关键条款或问题条件不足）。"

    return {
        "final_clause_ids": final_clause_ids,
        "final_applicability_label": final_appl,
        "final_conditions": cond_uniq,
        "final_answer": final_answer,
        "citations": cites_uniq,
        "accepted_experts": [e.get("agent_id") for e in expert_objs],
        "rejected_points": [],
        "decision_notes": "aggregated_without_arbiter"
    }

def main() -> None:
    cfg = json.loads(read_text("configs/ma_config.json"))

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    dash_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not openai_key or not dash_key:
        raise RuntimeError("Missing OPENAI_API_KEY or DASHSCOPE_API_KEY in environment.")

    in_path = "dev200/inputs/smoke5_with_candidates.jsonl"
    items = read_jsonl(in_path)

    ts = now_ts()
    out_agents = f"dev200/outputs/smoke5_ma6_qwen3max_experts_agents.{ts}.jsonl"
    out_final  = f"dev200/outputs/smoke5_ma6_qwen3max_gpt4o_arbiter_final.{ts}.jsonl"
    out_adj    = f"dev200/outputs/smoke5_ma6_gpt4o_adjudication.{ts}.jsonl"
    log_path   = f"dev200/logs/ma_smoke5.{ts}.log"

    # init files
    for p in [out_agents, out_final, out_adj, log_path]:
        open(p, "w", encoding="utf-8").close()

    experts = cfg["experts"]
    expert_model_primary = cfg.get("expert_model_primary", "qwen3-max")
    expert_model_fallbacks = cfg.get("expert_model_fallbacks", [])
    arbiter_model = cfg.get("arbiter_model", "gpt-4o")

    # prompts
    expert_prompts: Dict[str, str] = {}
    for e in experts:
        expert_prompts[e["agent_id"]] = read_text(e["prompt_file"])
    arbiter_prompt = read_text(cfg["arbiter_prompt_file"])

    top_k = int(cfg.get("candidates_top_k", 30))
    trig_cfg = cfg.get("arbitration_trigger", {})
    trigger_mode = (os.getenv("MA_TRIGGER_MODE") or "current").strip().lower()
    if trigger_mode in ("none","off","majority_only"):
        trig_cfg = {"clause_disagree": False, "applicability_disagree": False, "evidence_missing": False}
    elif trigger_mode not in ("", "current", "default", "all"):
        _m = trigger_mode
        # normalize aliases
        if _m in ("clause","clause_only"):
            _m = "clause_disagree_only"
        if _m in ("evidence","evidence_only"):
            _m = "evidence_missing_only"
        if _m in ("clause+evidence","clause_evidence","clause_or_evidence_missing"):
            _m = "clause_or_evidence"
        if _m in ("applicability","appl","applicability_disagree_only"):
            _m = "applicability_only"
    
        trig_cfg = dict(trig_cfg) if isinstance(trig_cfg, dict) else {}
        trig_cfg["clause_disagree"] = _m in ("clause_disagree_only","clause_or_evidence")
        trig_cfg["applicability_disagree"] = _m in ("applicability_only",)
        trig_cfg["evidence_missing"] = _m in ("evidence_missing_only","clause_or_evidence")
        trigger_mode = _m

    log_line(log_path, f"[START] smoke5 ts={ts}")
    log_line(log_path, f"[BASE] dashscope={DASHSCOPE_BASE_URL} openai={OPENAI_BASE_URL}")
    log_line(log_path, f"[INPUT] {in_path}")
    log_line(log_path, f"[OUT] agents={out_agents}")
    log_line(log_path, f"[OUT] final={out_final}")
    log_line(log_path, f"[OUT] adjudication={out_adj}")

    for idx, row in enumerate(items, 1):
        qid = pick_field(row, ["query_id", "qid", "id", "queryId"]) or f"smoke5_{idx}"
        query = pick_field(row, ["query", "question", "q", "text"]) or ""

        raw_cands = pick_field(row, ["candidates", "with_candidates", "candidate_clauses", "cand", "topk", "hits"])
        cands = normalize_candidates(raw_cands, top_k=top_k)

        if not cands and isinstance(row, dict):
            for k in ["candidates", "cand", "top_k_candidates", "retrieval_candidates"]:
                if k in row:
                    cands = normalize_candidates(row[k], top_k=top_k)
                    if cands:
                        break

        cand_block = build_candidates_block(cands, max_chars_each=1200)
        log_line(log_path, f"\n[Q{idx}] qid={qid} cand_n={len(cands)}")

        expert_outputs: List[Dict[str, Any]] = []
        for e in experts:
            aid = e["agent_id"]
            sys_prompt = expert_prompts[aid]
            user_msg = (
                f"【用户问题】\n{query}\n\n"
                f"【候选条款（Top-{min(len(cands), top_k)}）】\n{cand_block}\n\n"
                f"请严格按你的输出JSON schema返回。"
            )

            models_to_try = [expert_model_primary] + list(expert_model_fallbacks)
            content = None
            last_err = None

            for m in models_to_try:
                try:
                    content = chat_completions(
                        base_url=DASHSCOPE_BASE_URL,
                        api_key=dash_key,
                        model=m,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.0,
                        max_tokens=900,
                        timeout=180,
                    )
                    break
                except Exception as ex:
                    last_err = ex
                    log_line(log_path, f"[WARN] expert {aid} model={m} failed: {repr(ex)}")
                    time.sleep(1.0)

            if content is None:
                raise RuntimeError(f"Expert {aid} failed all models. last_err={last_err}")

            parsed = None
            for attempt in range(2):
                try:
                    parsed = extract_json_obj(content)
                    break
                except Exception as ex:
                    log_line(log_path, f"[WARN] parse expert {aid} attempt={attempt} err={repr(ex)} raw={content[:260]}")
                    content = chat_completions(
                        base_url=DASHSCOPE_BASE_URL,
                        api_key=dash_key,
                        model=expert_model_primary,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_msg + "\n\n再次强调：只输出一个JSON对象，不要任何多余字符。"},
                        ],
                        temperature=0.0,
                        max_tokens=900,
                        timeout=180,
                    )

            if parsed is None:
                raise RuntimeError(f"Cannot parse expert {aid} output.")

            parsed["agent_id"] = aid
            parsed.setdefault("discipline_zh", e.get("discipline_zh"))
            expert_outputs.append(parsed)

        append_jsonl(out_agents, {
            "query_id": qid,
            "query": query,
            "candidate_count": len(cands),
            "experts": expert_outputs
        })

        trigger, diag = disagreement_and_trigger(expert_outputs, trig_cfg)
        # [MA_TRIGGER_MODE_PATCH]
        trigger_mode = (os.getenv("MA_TRIGGER_MODE") or "current").strip().lower()
        if isinstance(diag, dict):
            diag["trigger_mode"] = trigger_mode
        if trigger_mode == "majority_only":
            trigger = False
        elif trigger_mode == "clause_disagree_only":
            trigger = bool((diag or {}).get("clause_disagree"))
        elif trigger_mode == "evidence_missing_only":
            trigger = bool((diag or {}).get("evidence_missing"))
        elif trigger_mode == "applicability_only":
            trigger = bool((diag or {}).get("applicability_disagree"))
        elif trigger_mode == "clause_or_evidence":
            trigger = bool((diag or {}).get("clause_disagree") or (diag or {}).get("evidence_missing"))
        else:
            # current/default: keep original trigger
            pass

        log_line(log_path, f"[Q{idx}] trigger={trigger} reasons={diag.get('trigger_reasons')}")

        if trigger:
            experts_json = json.dumps(expert_outputs, ensure_ascii=False)
            arb_user = (
                f"【用户问题】\n{query}\n\n"
                f"【候选条款（Top-{min(len(cands), top_k)}）】\n{cand_block}\n\n"
                f"【6位专家意见JSON数组】\n{experts_json}\n\n"
                f"请严格按你输出JSON schema返回裁决。"
            )

            content = chat_completions(
                base_url=OPENAI_BASE_URL,
                api_key=openai_key,
                model=arbiter_model,
                messages=[
                    {"role": "system", "content": arbiter_prompt},
                    {"role": "user", "content": arb_user},
                ],
                temperature=0.0,
                max_tokens=1400,
                timeout=180,
            )

            arb = None
            for attempt in range(2):
                try:
                    arb = extract_json_obj(content)
                    break
                except Exception as ex:
                    log_line(log_path, f"[WARN] parse arbiter attempt={attempt} err={repr(ex)} raw={content[:260]}")
                    content = chat_completions(
                        base_url=OPENAI_BASE_URL,
                        api_key=openai_key,
                        model=arbiter_model,
                        messages=[
                            {"role": "system", "content": arbiter_prompt},
                            {"role": "user", "content": arb_user + "\n\n再次强调：只输出一个JSON对象，不要任何多余字符。"},
                        ],
                        temperature=0.0,
                        max_tokens=1400,
                        timeout=180,
                    )

            if arb is None:
                raise RuntimeError("Cannot parse arbiter output.")

            append_jsonl(out_adj, {
                "query_id": qid,
                "query": query,
                "trigger_diag": diag,
                "arbiter": arb
            })

            final = arb
            final.setdefault("decision_notes", "")
            final["decision_notes"] = (final.get("decision_notes", "") + " | gpt4o_arbiter").strip()
        else:
            final = aggregate_without_arbiter(expert_outputs)

        append_jsonl(out_final, {
            "query_id": qid,
            "query": query,
            "trigger_mode": trigger_mode,
            "triggered_arbitration": trigger,
            "trigger_diag": diag,
            "final": final
        })

        time.sleep(0.2)

    log_line(log_path, "\n[END] smoke5 done.")
    print("DONE")
    print("agents:", out_agents)
    print("final :", out_final)
    print("adj   :", out_adj)
    print("log   :", log_path)

if __name__ == "__main__":
    main()
