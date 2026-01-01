import os, json
from collections import Counter

TAG = os.getenv("MA_RUN_TAG")
if not TAG:
    raise SystemExit("Please export MA_RUN_TAG first.")

AGENTS = f"test100/outputs/{TAG}.agents.jsonl"
FINAL  = f"test100/outputs/{TAG}.final.jsonl"

LABELS_CANDIDATES = [
    "/home/RagConstructionAssistant/experiments/retrieval/data/test_labels.json",
    "/home/RagConstructionAssistant/experiments/retrieval/data/test_labels.jsonl",
]

def load_labels():
    path = None
    for p in LABELS_CANDIDATES:
        if os.path.exists(p):
            path = p
            break
    if not path:
        raise SystemExit(f"Cannot find test_labels in candidates: {LABELS_CANDIDATES}")

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
    gold = {}

    def to_set(x):
        if x is None:
            return set()
        if isinstance(x, (str,int)):
            return {str(x)}
        if isinstance(x, list):
            return {str(i) for i in x if i is not None and str(i).strip() != ""}
        if isinstance(x, dict):
            for k in ["clause_id","gold_clause_id","label_clause_id","answer_clause_id","target_clause_id"]:
                if k in x:
                    return to_set(x[k])
            for k in ["clause_ids","gold_clause_ids","label_clause_ids","answers","labels"]:
                if k in x:
                    return to_set(x[k])
        return set()

    if isinstance(data, dict):
        for qid, v in data.items():
            gold[str(qid)] = to_set(v)
        return gold

    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict):
                continue
            qid = it.get("query_id") or it.get("qid") or it.get("id") or it.get("queryId")
            if qid is None:
                continue
            s = to_set(it)
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
    for e in experts:
        sel = e.get("selected_clause_ids") or []
        if sel:
            tops.append(str(sel[0]))
    if not tops:
        return None
    return Counter(tops).most_common(1)[0][0]

def final_clause_id(final_obj):
    if not isinstance(final_obj, dict):
        return None
    if final_obj.get("final_clause_ids"):
        return str(final_obj["final_clause_ids"][0])
    if final_obj.get("final_clause_id"):
        return str(final_obj["final_clause_id"])
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

label_path, label_data = load_labels()
gold = extract_gold_map(label_data)

agents_rows = read_jsonl(AGENTS)
final_rows  = read_jsonl(FINAL)
agents_by_qid = {str(r["query_id"]): r for r in agents_rows}

overall = {"n":0, "maj_hit1":0, "arb_hit1":0}
triggered = {"n":0, "maj_hit1":0, "arb_hit1":0}
changed = {"n":0, "maj_hit1":0, "arb_hit1":0}  # arbiter != majority
missing_gold = 0
missing_majority = 0
missing_final = 0

for r in final_rows:
    qid = str(r["query_id"])
    gset = gold.get(qid, set())
    if not gset:
        missing_gold += 1
        continue

    experts = (agents_by_qid.get(qid) or {}).get("experts") or []
    maj = majority_top_clause(experts)
    arb = final_clause_id(r.get("final") or {})

    if maj is None:
        missing_majority += 1
    if arb is None:
        missing_final += 1

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

def pct(x): 
    return (x[0]/x[1] if x[1] else 0.0)

print("LABEL_FILE:", label_path)
print("EVAL_N (with gold):", overall["n"], "missing_gold:", missing_gold)
print("missing_majority:", missing_majority, "missing_final:", missing_final)

print("\n[OVERALL]")
print("  majority_hit@1:", round(pct((overall["maj_hit1"],overall["n"])),4), f'({overall["maj_hit1"]}/{overall["n"]})')
print("  arbiter_hit@1 :", round(pct((overall["arb_hit1"],overall["n"])),4), f'({overall["arb_hit1"]}/{overall["n"]})')
print("  delta         :", round(pct((overall["arb_hit1"],overall["n"]))-pct((overall["maj_hit1"],overall["n"])),4))

print("\n[TRIGGERED subset]")
print("  n:", triggered["n"])
print("  majority_hit@1:", round(pct((triggered["maj_hit1"],triggered["n"])),4), f'({triggered["maj_hit1"]}/{triggered["n"]})')
print("  arbiter_hit@1 :", round(pct((triggered["arb_hit1"],triggered["n"])),4), f'({triggered["arb_hit1"]}/{triggered["n"]})')
print("  delta         :", round(pct((triggered["arb_hit1"],triggered["n"]))-pct((triggered["maj_hit1"],triggered["n"])),4))

print("\n[ARBITER != MAJORITY subset]")
print("  n:", changed["n"])
print("  majority_hit@1:", round(pct((changed["maj_hit1"],changed["n"])),4), f'({changed["maj_hit1"]}/{changed["n"]})')
print("  arbiter_hit@1 :", round(pct((changed["arb_hit1"],changed["n"])),4), f'({changed["arb_hit1"]}/{changed["n"]})')
print("  delta         :", round(pct((changed["arb_hit1"],changed["n"]))-pct((changed["maj_hit1"],changed["n"])),4))
