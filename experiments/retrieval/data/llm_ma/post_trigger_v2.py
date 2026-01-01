import os, json, shutil, time
from pathlib import Path
from collections import Counter

SPLIT = os.environ["MA_SPLIT"].strip()
BASE  = os.environ["MA_BASE_TAG"].strip()
STRAT = os.environ["MA_STRAT"].strip()
LABEL = Path(os.environ["MA_LABEL_PATH"].strip())

base_agents = Path(f"{SPLIT}/outputs/{BASE}.agents.jsonl")
base_adj    = Path(f"{SPLIT}/outputs/{BASE}.adjudication.jsonl")
base_final  = Path(f"{SPLIT}/outputs/{BASE}.final.jsonl")

for p in (base_agents, base_final, LABEL):
    if not p.exists():
        raise SystemExit(f"[FAIL] missing file: {p}")

def to_bool(x):
    if isinstance(x, bool): return x
    if x is None: return False
    if isinstance(x, (int, float)): return x != 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true","1","yes","y","t"): return True
        if s in ("false","0","no","n","f",""): return False
        return True
    return bool(x)

gold_raw = json.load(open(LABEL, "r", encoding="utf-8"))

def gold_set(v):
    if v is None: return set()
    if isinstance(v, list):
        out = set()
        for x in v:
            s = str(x).strip()
            out.add(str(int(s)) if s.isdigit() else s)
        return out
    s = str(v).strip()
    return {str(int(s)) if s.isdigit() else s}

gold = {str(k): gold_set(v) for k, v in gold_raw.items()}

def tie_key(x):
    s = str(x).strip()
    return (0, int(s)) if s.isdigit() else (1, s)

# 1) majority from agents
qid2maj = {}
with open(base_agents, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        qid = str(r.get("query_id") or r.get("qid") or r.get("id"))
        votes = []
        for e in (r.get("experts") or []):
            sel = e.get("selected_clause_ids") or []
            if sel:
                votes.append(str(sel[0]).strip())
        if votes:
            c = Counter(votes)
            best = max(c.values())
            tied = [k for k, v in c.items() if v == best]
            qid2maj[qid] = sorted(tied, key=tie_key)[0]
        else:
            qid2maj[qid] = "ABSTAIN"

# 2) trigger set from final (trigger_diag 在 final 里!)
use_arb = set()
arb_missing = set()
triggered_in_base = 0

with open(base_final, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        qid = str(r.get("query_id") or r.get("qid") or r.get("id"))
        base_trig = to_bool(r.get("triggered_arbitration"))
        if base_trig:
            triggered_in_base += 1

        diag = r.get("trigger_diag") or {}

        if STRAT == "majority_only":
            want = False
        elif STRAT == "current_trigger_all":
            want = base_trig
        elif STRAT == "clause_disagree_only":
            want = to_bool(diag.get("clause_disagree"))
        elif STRAT == "evidence_missing_only":
            want = to_bool(diag.get("evidence_missing"))
        elif STRAT == "clause_or_evidence":
            want = to_bool(diag.get("clause_disagree")) or to_bool(diag.get("evidence_missing"))
        elif STRAT == "applicability_only":
            want = to_bool(diag.get("applicability_disagree"))
        else:
            raise SystemExit(f"[FAIL] unknown STRAT={STRAT}")

        if want:
            if base_trig:
                use_arb.add(qid)
            else:
                arb_missing.add(qid)

post_tag = f"{BASE}.POST_{STRAT}.v2"
post_agents = Path(f"{SPLIT}/outputs/{post_tag}.agents.jsonl")
post_adj    = Path(f"{SPLIT}/outputs/{post_tag}.adjudication.jsonl")
post_final  = Path(f"{SPLIT}/outputs/{post_tag}.final.jsonl")

ts = time.strftime("%Y%m%d_%H%M%S")
for p in (post_agents, post_adj, post_final):
    if p.exists():
        shutil.copyfile(p, Path(str(p) + f".bak.{ts}"))

# post_agents: copy
shutil.copyfile(base_agents, post_agents)

# post_adj: filter by use_arb
adj_in = 0
adj_out = 0
with open(post_adj, "w", encoding="utf-8") as fout:
    if base_adj.exists():
        with open(base_adj, "r", encoding="utf-8") as fin:
            for line in fin:
                adj_in += 1
                rr = json.loads(line)
                qid = str(rr.get("query_id") or rr.get("qid") or rr.get("id"))
                if qid in use_arb:
                    fout.write(line if line.endswith("\n") else line + "\n")
                    adj_out += 1

# post_final: keep arbiter for use_arb else overwrite with majority
n = 0
hit = 0
arb_used = 0

with open(post_final, "w", encoding="utf-8") as fout:
    with open(base_final, "r", encoding="utf-8") as fin:
        for line in fin:
            r = json.loads(line)
            qid = str(r.get("query_id") or r.get("qid") or r.get("id"))
            gs = gold.get(qid, set())
            if not gs:
                continue
            n += 1

            if qid in use_arb:
                arb_used += 1
                top1 = (r.get("final") or {}).get("final_clause_ids") or []
                t = str(top1[0]).strip() if top1 else "ABSTAIN"
                if t in gs:
                    hit += 1
            else:
                r["triggered_arbitration"] = False
                maj = qid2maj.get(qid, "ABSTAIN")
                if "final" not in r or not isinstance(r["final"], dict):
                    r["final"] = {}
                r["final"]["final_clause_ids"] = [maj] if maj != "ABSTAIN" else []
                if maj in gs:
                    hit += 1

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

Path("results/last_post_tag.txt").write_text(post_tag, encoding="utf-8")

print(f"SPLIT={SPLIT} BASE={BASE} STRAT={STRAT}")
print(f"triggered_in_base_final={triggered_in_base} want_use_arb={len(use_arb)} arb_missing={len(arb_missing)}")
print(f"[OK] POST_TAG={post_tag}")
print(f"arb_used={arb_used} hit@1(post_final_top1)={hit/n:.4f} ({hit}/{n})")
print(f"FILES: {post_agents} | {post_adj} (adj_in={adj_in} adj_out={adj_out}) | {post_final}")
