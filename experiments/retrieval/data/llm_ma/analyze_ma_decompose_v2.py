import os, json
from collections import Counter

def read_jsonl(p):
    rows=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_labels(path):
    data=json.load(open(path,"r",encoding="utf-8"))
    gold={}
    if isinstance(data, dict):
        for qid,v in data.items():
            if isinstance(v, list):
                gold[str(qid)] = {str(x) for x in v}
            else:
                gold[str(qid)] = {str(v)}
    elif isinstance(data, list):
        for it in data:
            qid=it.get("query_id") or it.get("qid") or it.get("id")
            if qid is None:
                continue
            v = it.get("gold") or it.get("gold_clause_id") or it.get("clause_id") or it.get("label")
            if isinstance(v, list):
                gold[str(qid)] = {str(x) for x in v}
            elif v is not None:
                gold[str(qid)] = {str(v)}
    return gold

def extract_candidate_ids(obj):
    """
    robust extraction from a record that already has candidates.
    supports list[str/int] or list[dict] where dict may contain clause_id/id.
    """
    keys = ["candidates","candidate_clauses","candidate_clause_ids","cands","retrieval_candidates","dense_candidates","bm25_candidates"]
    cand = []
    for k in keys:
        if k in obj and obj[k]:
            cand = obj[k]
            break
    s=set()
    if isinstance(cand, list):
        for x in cand:
            if isinstance(x, dict):
                cid = x.get("clause_id") or x.get("id") or x.get("candidate_clause_id")
                if cid is not None:
                    s.add(str(cid))
            else:
                s.add(str(x))
    return s

def load_candidate_map(split):
    """
    candidates are guaranteed in *with_candidates.jsonl* (inputs), not necessarily in final outputs.
    """
    if split == "dev200":
        in_path = "dev200/inputs/dev200_with_candidates.jsonl"
    elif split == "test100":
        in_path = "test100/inputs/test100_with_candidates.jsonl"
    else:
        raise SystemExit(f"Unknown split: {split}")

    rows = read_jsonl(in_path)
    mp={}
    for r in rows:
        qid = r.get("query_id")
        if qid is None:
            continue
        mp[str(qid)] = extract_candidate_ids(r)
    return in_path, mp

def majority_top1_strict(experts):
    """
    strict majority top1 on experts' selected_clause_ids[0]
    tie -> None
    """
    tops=[]
    for e in experts:
        sel=e.get("selected_clause_ids") or []
        if sel:
            tops.append(str(sel[0]))
    if not tops:
        return None
    ctr = Counter(tops).most_common()
    if len(ctr) >= 2 and ctr[0][1] == ctr[1][1]:
        return None
    return ctr[0][0]

def final_top1(final_obj):
    if not isinstance(final_obj, dict):
        return None
    if final_obj.get("final_clause_ids"):
        return str(final_obj["final_clause_ids"][0])
    if final_obj.get("final_clause_id"):
        return str(final_obj["final_clause_id"])
    # fallbacks
    for k in ["selected_clause_id","selected_clause_ids","clause_id"]:
        if k in final_obj:
            v = final_obj[k]
            if isinstance(v, list) and v:
                return str(v[0])
            if isinstance(v, (str,int)):
                return str(v)
    return None

def get_expert_union(experts):
    top1=set()
    anyset=set()
    for e in experts:
        sel=e.get("selected_clause_ids") or []
        if sel:
            top1.add(str(sel[0]))
            for x in sel:
                anyset.add(str(x))
    return top1, anyset

def run(split, run_tag, label_path):
    agents_path=f"{split}/outputs/{run_tag}.agents.jsonl"
    final_path =f"{split}/outputs/{run_tag}.final.jsonl"
    agents=read_jsonl(agents_path)
    finals=read_jsonl(final_path)
    gold=load_labels(label_path)

    cand_src, cand_map = load_candidate_map(split)

    agents_by_qid={str(r["query_id"]): r for r in agents}

    n=0
    miss_gold=0
    miss_majority=[]
    miss_final=[]
    ub_cand=0
    ub_expert_top1=0
    ub_expert_any=0

    maj_hit=0
    arb_hit=0

    help_n=0
    harm_n=0
    changed_n=0
    changed_help=0
    changed_harm=0

    for r in finals:
        qid=str(r["query_id"])
        g=gold.get(qid, set())
        if not g:
            miss_gold += 1
            continue
        n += 1

        cand = cand_map.get(qid, set())
        if cand and (g & cand):
            ub_cand += 1

        experts=(agents_by_qid.get(qid) or {}).get("experts") or []
        top1_set, any_set = get_expert_union(experts)

        if g & top1_set:
            ub_expert_top1 += 1
        if g & any_set:
            ub_expert_any += 1

        maj = majority_top1_strict(experts)
        arb = final_top1(r.get("final") or {})

        if maj is None:
            miss_majority.append(qid)
        if arb is None:
            miss_final.append(qid)

        maj_ok = 1 if (maj is not None and str(maj) in g) else 0
        arb_ok = 1 if (arb is not None and str(arb) in g) else 0
        maj_hit += maj_ok
        arb_hit += arb_ok

        if maj_ok==0 and arb_ok==1:
            help_n += 1
        if maj_ok==1 and arb_ok==0:
            harm_n += 1

        if maj is not None and arb is not None and maj != arb:
            changed_n += 1
            if maj_ok==0 and arb_ok==1:
                changed_help += 1
            if maj_ok==1 and arb_ok==0:
                changed_harm += 1

    print(f"SPLIT={split} RUN_TAG={run_tag}")
    print(f"labels={label_path}")
    print(f"candidates_from={cand_src}")
    print(f"n={n} miss_gold={miss_gold}")
    print(f"hit@1 majority={maj_hit/n:.4f} ({maj_hit}/{n}) arbiter={arb_hit/n:.4f} ({arb_hit}/{n}) delta={(arb_hit-maj_hit)/n:.4f}")
    print(f"upper_bound gold_in_candidates={ub_cand/n:.4f} ({ub_cand}/{n})")
    print(f"upper_bound gold_in_expert_top1={ub_expert_top1/n:.4f} ({ub_expert_top1}/{n})")
    print(f"upper_bound gold_in_expert_any ={ub_expert_any/n:.4f} ({ub_expert_any}/{n})")
    print(f"helpful (maj wrong -> arb right)={help_n} | harmful (maj right -> arb wrong)={harm_n}")
    print(f"changed_n={changed_n} | changed_help={changed_help} | changed_harm={changed_harm}")
    print(f"missing_majority_qids={miss_majority[:20]}")
    print(f"missing_final_qids={miss_final[:20]}")

if __name__ == "__main__":
    split=os.environ.get("MA_SPLIT")
    tag=os.environ.get("MA_RUN_TAG")
    label=os.environ.get("MA_LABEL_PATH")
    if not (split and tag and label):
        raise SystemExit("Need env: MA_SPLIT, MA_RUN_TAG, MA_LABEL_PATH")
    run(split, tag, label)
