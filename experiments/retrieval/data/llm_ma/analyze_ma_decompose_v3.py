import os, json, re
from collections import Counter

# 直接复用 run_ma_dev200.py 的候选解析逻辑（与你运行仲裁时一致）
from run_ma_dev200 import split_question_and_candidates, parse_candidates_from_text

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

def find_clause_id_in_obj(obj):
    """从 dict 里尽可能鲁棒地找 clause_id"""
    if not isinstance(obj, dict):
        return None

    # 常见字段
    for k in ["clause_id","clauseId","clauseID","id","cid","clause_id_str"]:
        if k in obj and obj[k] is not None:
            return str(obj[k])

    # meta 里可能有
    meta = obj.get("meta")
    if isinstance(meta, dict):
        for k in ["clause_id","clauseId","clauseID","id","cid"]:
            if k in meta and meta[k] is not None:
                return str(meta[k])

    # 兜底：找 key 名包含 clause+id
    for k,v in obj.items():
        kk = str(k).lower()
        if ("clause" in kk and "id" in kk) and v is not None:
            return str(v)

    return None

def build_candidate_map(split, top_k=30):
    if split == "dev200":
        in_path = "dev200/inputs/dev200_with_candidates.jsonl"
    elif split == "test100":
        in_path = "test100/inputs/test100_with_candidates.jsonl"
    else:
        raise SystemExit(f"Unknown split: {split}")

    mp={}
    rows=read_jsonl(in_path)
    for r in rows:
        qid = r.get("query_id") or r.get("qid") or r.get("id")
        if qid is None:
            continue
        qid = str(qid)

        qraw = r.get("query") or r.get("raw_query") or ""
        _, cand_text = split_question_and_candidates(qraw)

        cids=set()
        if cand_text:
            cands = parse_candidates_from_text(cand_text, top_k=top_k) or []
            for c in cands:
                cid = find_clause_id_in_obj(c)
                if cid is not None:
                    cids.add(cid)

        mp[qid]=cids
    return in_path, mp

def majority_top1_strict(experts):
    """严格多数：平票 -> None"""
    tops=[]
    for e in experts:
        sel=e.get("selected_clause_ids") or []
        if sel:
            tops.append(str(sel[0]))
    if not tops:
        return None
    ctr=Counter(tops).most_common()
    if len(ctr)>=2 and ctr[0][1]==ctr[1][1]:
        return None
    return ctr[0][0]

def final_top1(final_obj):
    if not isinstance(final_obj, dict):
        return None
    if final_obj.get("final_clause_ids"):
        return str(final_obj["final_clause_ids"][0])
    if final_obj.get("final_clause_id"):
        return str(final_obj["final_clause_id"])
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

    cand_src, cand_map = build_candidate_map(split, top_k=50)

    agents_by_qid={str(r["query_id"]): r for r in agents}

    n=0
    ub_cand=0
    ub_expert_top1=0
    ub_expert_any=0
    maj_hit=0
    arb_hit=0
    help_n=0
    harm_n=0
    miss_majority=[]
    miss_final=[]

    cand_sizes=[len(v) for v in cand_map.values() if isinstance(v,set)]

    for r in finals:
        qid=str(r["query_id"])
        g=gold.get(qid, set())
        if not g:
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

        maj_ok = 1 if (maj is not None and maj in g) else 0
        arb_ok = 1 if (arb is not None and arb in g) else 0
        maj_hit += maj_ok
        arb_hit += arb_ok

        if maj_ok==0 and arb_ok==1:
            help_n += 1
        if maj_ok==1 and arb_ok==0:
            harm_n += 1

    print(f"SPLIT={split} RUN_TAG={run_tag}")
    print(f"labels={label_path}")
    print(f"candidates_from={cand_src}")
    if cand_sizes:
        print(f"cand_size min/avg/max = {min(cand_sizes)} / {sum(cand_sizes)/len(cand_sizes):.2f} / {max(cand_sizes)}")
    else:
        print("cand_size min/avg/max = (EMPTY)  <-- parse failed, need inspect query format")

    print(f"n={n}")
    print(f"hit@1 majority={maj_hit/n:.4f} ({maj_hit}/{n}) arbiter={arb_hit/n:.4f} ({arb_hit}/{n}) delta={(arb_hit-maj_hit)/n:.4f}")
    print(f"upper_bound gold_in_candidates={ub_cand/n:.4f} ({ub_cand}/{n})")
    print(f"upper_bound gold_in_expert_top1={ub_expert_top1/n:.4f} ({ub_expert_top1}/{n})")
    print(f"upper_bound gold_in_expert_any ={ub_expert_any/n:.4f} ({ub_expert_any}/{n})")
    print(f"helpful={help_n} harmful={harm_n}")
    print(f"missing_majority_qids(sample)={miss_majority[:15]}")
    print(f"missing_final_qids(sample)={miss_final[:15]}")

if __name__ == "__main__":
    split=os.environ.get("MA_SPLIT")
    tag=os.environ.get("MA_RUN_TAG")
    label=os.environ.get("MA_LABEL_PATH")
    if not (split and tag and label):
        raise SystemExit("Need env: MA_SPLIT, MA_RUN_TAG, MA_LABEL_PATH")
    run(split, tag, label)
