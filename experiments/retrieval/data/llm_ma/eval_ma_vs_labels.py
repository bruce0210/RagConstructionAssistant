import os, json
from collections import Counter, defaultdict

TAG = os.getenv("MA_RUN_TAG")

SPLIT = (os.getenv("MA_SPLIT") or ("test100" if (TAG or "").startswith("test100_") else "dev200")).strip()
LABEL_FILE = (os.getenv("MA_LABEL_PATH") or os.getenv("MA_LABEL_FILE") or ("/home/RagConstructionAssistant/experiments/retrieval/data/test_labels.json" if SPLIT=="test100" else "/home/RagConstructionAssistant/experiments/retrieval/data/dev_labels.json")).strip()
if not TAG:
    raise SystemExit("Please export MA_RUN_TAG first.")

AGENTS = f"{SPLIT}/outputs/{TAG}.agents.jsonl"
FINAL  = f"{SPLIT}/outputs/{TAG}.final.jsonl"
# likely path (from your retrieval experiments)
LABELS_CANDIDATES = [
    "/home/RagConstructionAssistant/experiments/retrieval/data/dev_labels.json",
    "/home/RagConstructionAssistant/experiments/retrieval/data/dev_labels.jsonl",
    "dev200/inputs/dev_labels.json",
    "dev200/inputs/dev_labels.jsonl",
]

def load_labels():
    path = None
    for p in LABELS_CANDIDATES:
        if os.path.exists(p):
            path = p
            break
    if not path:
        raise SystemExit(f"Cannot find dev_labels in candidates: {LABELS_CANDIDATES}")

    # json or jsonl
    if path.endswith(".jsonl"):
        rows=[]
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    rows.append(json.loads(line))
        data = rows
    else:
        data = json.load(open(path,"r",encoding="utf-8"))
    return path, data

def extract_gold_map(data):
    """
    Build qid -> set(gold_clause_ids)
    Supports common shapes:
      - dict: {qid: clause_id or [clause_ids] or {...}}
      - list[dict]: each has query_id/qid and clause_id/label/...
    """
    gold = {}

    def to_set(x):
        if x is None:
            return set()
        if isinstance(x, (str,int)):
            return {str(x)}
        if isinstance(x, list):
            return {str(i) for i in x if i is not None and str(i).strip() != ""}
        if isinstance(x, dict):
            # try common keys inside dict
            for k in ["clause_id","gold_clause_id","label_clause_id","answer_clause_id","target_clause_id"]:
                if k in x:
                    return to_set(x[k])
            for k in ["clause_ids","gold_clause_ids","label_clause_ids","answers","labels"]:
                if k in x:
                    return to_set(x[k])
        return set()

    if isinstance(data, dict):
        for qid, v in data.items():
            # v might be id/list/dict
            s = to_set(v)
            if not s and isinstance(v, dict):
                # maybe nested with query_id
                s = to_set(v)
            gold[str(qid)] = s
        return gold

    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict):
                continue
            qid = it.get("query_id") or it.get("qid") or it.get("id") or it.get("queryId")
            if qid is None:
                continue
            s = to_set(it)
            # if still empty, try keys explicitly
            if not s:
                for k in ["clause_id","gold_clause_id","label_clause_id","answer_clause_id","target_clause_id"]:
                    if k in it:
                        s = to_set(it[k]); break
            gold[str(qid)] = s
        return gold

    raise SystemExit("Unsupported label file format.")

def read_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def majority_top_clause(experts):
    tops=[]
    conf={}
    for e in experts:
        sel = e.get("selected_clause_ids") or []
        if sel:
            cid=str(sel[0])
            tops.append(cid)
            try:
                conf[cid] = max(conf.get(cid, 0.0), float(e.get("confidence") or 0.0))
            except:
                conf[cid] = conf.get(cid, 0.0)
    if not tops:
        return None
    c = Counter(tops)
    best = max(c.values())
    tied = [k for k,v in c.items() if v==best]
    if len(tied)==1:
        return tied[0]
    tied.sort(key=lambda x: conf.get(x,0.0), reverse=True)
    return tied[0]

def final_clause_id(final_obj):
    if not isinstance(final_obj, dict):
        return None
    if "final_clause_ids" in final_obj and final_obj["final_clause_ids"]:
        return str(final_obj["final_clause_ids"][0])
    if "final_clause_id" in final_obj and final_obj["final_clause_id"]:
        return str(final_obj["final_clause_id"])
    # fallback: sometimes arbiter might output selected_clause_id
    for k in ["selected_clause_id","selected_clause_ids","clause_id"]:
        if k in final_obj:
            v = final_obj[k]
            if isinstance(v, list) and v:
                return str(v[0])
            if isinstance(v, (str,int)):
                return str(v)
    return None

def hit1(pred, gold_set):
    if pred is None:
        return 0
    return 1 if str(pred) in gold_set else 0

# --- load data ---
label_path, label_data = load_labels()
gold = extract_gold_map(label_data)

agents_rows = read_jsonl(AGENTS)
final_rows  = read_jsonl(FINAL)

agents_by_qid = {str(r["query_id"]): r for r in agents_rows}

# --- metrics ---
overall = {"n":0, "maj_hit1":0, "arb_hit1":0}
triggered = {"n":0, "maj_hit1":0, "arb_hit1":0}
changed = {"n":0, "maj_hit1":0, "arb_hit1":0}  # arbiter != majority

missing_gold = 0

for r in final_rows:
    qid = str(r["query_id"])
    gset = gold.get(qid, set())
    if not gset:
        missing_gold += 1
        continue

    experts = (agents_by_qid.get(qid) or {}).get("experts") or []
    maj = majority_top_clause(experts)
    arb = final_clause_id(r.get("final") or {})

    overall["n"] += 1
    overall["maj_hit1"] += hit1(maj, gset)
    overall["arb_hit1"] += hit1(arb, gset)

    if r.get("triggered_arbitration"):
        triggered["n"] += 1
        triggered["maj_hit1"] += hit1(maj, gset)
        triggered["arb_hit1"] += hit1(arb, gset)

    if maj is not None and arb is not None and maj != arb:
        changed["n"] += 1
        changed["maj_hit1"] += hit1(maj, gset)
        changed["arb_hit1"] += hit1(arb, gset)

def rate(x):
    return (x["arb_hit1"]/x["n"] if x["n"] else 0.0,
            x["maj_hit1"]/x["n"] if x["n"] else 0.0)

arb_o, maj_o = rate(overall)
arb_t, maj_t = rate(triggered)
arb_c, maj_c = rate(changed)

print("LABEL_FILE:", label_path)
print("EVAL_N (with gold):", overall["n"], "missing_gold:", missing_gold)

print("\n[OVERALL]")
print("  majority_hit@1:", round(maj_o,4), f"({overall['maj_hit1']}/{overall['n']})")
print("  arbiter_hit@1 :", round(arb_o,4), f"({overall['arb_hit1']}/{overall['n']})")
print("  delta         :", round(arb_o-maj_o,4))

print("\n[TRIGGERED subset]")
print("  n:", triggered["n"])
print("  majority_hit@1:", round(maj_t,4), f"({triggered['maj_hit1']}/{triggered['n']})")
print("  arbiter_hit@1 :", round(arb_t,4), f"({triggered['arb_hit1']}/{triggered['n']})")
print("  delta         :", round(arb_t-maj_t,4))

print("\n[ARBITER != MAJORITY subset]")
print("  n:", changed["n"])
print("  majority_hit@1:", round(maj_c,4), f"({changed['maj_hit1']}/{changed['n']})")
print("  arbiter_hit@1 :", round(arb_c,4), f"({changed['arb_hit1']}/{changed['n']})")
print("  delta         :", round(arb_c-maj_c,4))
