import argparse, json, os
from collections import defaultdict
from statistics import mean, median
import pandas as pd

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_labels(path):
    raw = json.load(open(path, "r", encoding="utf-8"))
    gold = {}
    if isinstance(raw, dict):
        for qid, v in raw.items():
            if isinstance(v, str):
                gold[str(qid)] = [v]
            elif isinstance(v, list):
                gold[str(qid)] = [str(x) for x in v]
            elif isinstance(v, dict):
                for key in ["clause_id", "gold_clause_id", "answer", "label"]:
                    if key in v:
                        vv = v[key]
                        if isinstance(vv, str):
                            gold[str(qid)] = [vv]
                        elif isinstance(vv, list):
                            gold[str(qid)] = [str(x) for x in vv]
                        break
    elif isinstance(raw, list):
        for item in raw:
            qid = str(item.get("qid") or item.get("id") or "")
            if not qid:
                continue
            vv = item.get("clause_id") or item.get("gold_clause_id") or item.get("label") or item.get("answer")
            if isinstance(vv, str):
                gold[qid] = [vv]
            elif isinstance(vv, list):
                gold[qid] = [str(x) for x in vv]
    return gold

def load_query_meta(queries_jsonl):
    meta = {}
    for r in load_jsonl(queries_jsonl):
        qid = str(r.get("qid") or r.get("id") or r.get("query_id") or "")
        if not qid:
            continue
        meta[qid] = {
            "discipline": r.get("discipline", "UNKNOWN"),
            "type": r.get("type", "UNKNOWN"),
        }
    return meta

def pct(x, n):
    return 0.0 if n==0 else (x*100.0/n)

def quantile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    idx = int(round((len(s)-1)*q))
    return s[idx]

def cost_one(pricing, provider, model, usage):
    if not usage:
        return None
    pt = usage.get("prompt_tokens") or 0
    ct = usage.get("completion_tokens") or 0

    # OpenAI: USD / 1M tokens
    if provider == "openai":
        p = (pricing.get("openai_usd_per_1m") or {}).get(model)
        if not p:
            return None
        pin, pout = p.get("input"), p.get("output")
        if pin is None or pout is None:
            return None
        return (pt/1_000_000.0)*pin + (ct/1_000_000.0)*pout

    # DashScope(Qwen): USD / 1M tokens
    p = (pricing.get("dashscope_usd_per_1m") or {}).get(model)
    if not p:
        return None
    pin, pout = p.get("input"), p.get("output")
    if pin is None or pout is None:
        return None
    return (pt/1_000_000.0)*pin + (ct/1_000_000.0)*pout

def eval_rows(rows, gold_map):
    ks = [1,3,5,10]
    hit = {k:0 for k in ks}
    rr_sum = 0.0
    n = 0
    perq = []
    for r in rows:
        qid = str(r.get("qid",""))
        golds = gold_map.get(qid)
        if not golds:
            continue
        preds = r.get("pred_clause_ids") or []
        rank = None
        for i, pid in enumerate(preds, start=1):
            if pid in golds:
                rank = i
                break
        n += 1
        if rank is not None:
            rr_sum += 1.0/rank
            for k in ks:
                if rank <= k:
                    hit[k] += 1
        perq.append((qid, golds[0], preds[0] if preds else "", rank))
    mrr = 0.0 if n==0 else rr_sum/n
    out = {"n": n, "mrr": mrr}
    for k in ks:
        out[f"hit@{k}"] = 0.0 if n==0 else hit[k]/n
    return out, perq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pricing", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.pred)
    meta = load_query_meta(args.queries)
    gold = load_labels(args.labels)
    pricing = json.load(open(args.pricing,"r",encoding="utf-8"))

    # overall
    overall, perq_basic = eval_rows(rows, gold)

    lat = [r.get("latency_s") for r in rows if isinstance(r.get("latency_s"), (int,float))]
    overall["latency_mean_s"] = mean(lat) if lat else None
    overall["latency_median_s"] = median(lat) if lat else None
    overall["latency_p90_s"] = quantile(lat, 0.90)
    overall["latency_p95_s"] = quantile(lat, 0.95)

    # tokens & cost
    pt = [ (r.get("usage") or {}).get("prompt_tokens") for r in rows ]
    ct = [ (r.get("usage") or {}).get("completion_tokens") for r in rows ]
    ptv = [x for x in pt if isinstance(x,(int,float))]
    ctv = [x for x in ct if isinstance(x,(int,float))]
    overall["prompt_tokens_sum"] = int(sum(ptv)) if ptv else None
    overall["completion_tokens_sum"] = int(sum(ctv)) if ctv else None
    overall["total_tokens_sum"] = int(sum(ptv)+sum(ctv)) if ptv or ctv else None

    provider = rows[0].get("provider") if rows else None
    model = rows[0].get("model") if rows else None
    costs = []
    for r in rows:
        c = cost_one(pricing, r.get("provider"), r.get("model"), r.get("usage") or {})
        if c is not None:
            costs.append(c)
    overall["cost_sum"] = float(sum(costs)) if costs else None
    overall["cost_mean"] = float(mean(costs)) if costs else None
    overall["cost_unit"] = pricing.get("currency","USD")

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    with open(args.out_prefix + "_overall.json","w",encoding="utf-8") as f:
        json.dump({"provider":provider,"model":model,**overall}, f, ensure_ascii=False, indent=2)

    # per-query table
    perq_rows = []
    for r in rows:
        qid = str(r.get("qid",""))
        golds = gold.get(qid, [""])
        preds = r.get("pred_clause_ids") or []
        rank = None
        for i,pid in enumerate(preds, start=1):
            if golds and pid in golds:
                rank = i
                break
        perq_rows.append({
            "qid": qid,
            "discipline": meta.get(qid,{}).get("discipline","UNKNOWN"),
            "type": meta.get(qid,{}).get("type","UNKNOWN"),
            "gold": golds[0] if golds else "",
            "pred1": preds[0] if preds else "",
            "rank": rank,
            "hit@1": 1 if rank==1 else 0,
            "latency_s": r.get("latency_s"),
            "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
            "cost": cost_one(pricing, r.get("provider"), r.get("model"), r.get("usage") or {}),
            "error": r.get("error"),
        })
    df = pd.DataFrame(perq_rows)
    df.to_csv(args.out_prefix + "_per_query.csv", index=False)

    # group metrics
    out_rows = []
    for dim in ["discipline","type"]:
        for g, gdf in df.groupby(dim):
            n = len(gdf)
            hit1 = float(gdf["hit@1"].sum())/n if n else 0.0
            mrr = mean([ (1.0/r) if r and r>0 else 0.0 for r in gdf["rank"].tolist() ]) if n else 0.0
            out_rows.append({
                "group_by": dim,
                "group": g,
                "n": n,
                "hit@1": hit1,
                "mrr": mrr,
                "latency_mean_s": gdf["latency_s"].mean(),
                "cost_mean": gdf["cost"].mean(),
            })
    pd.DataFrame(out_rows).to_csv(args.out_prefix + "_group.csv", index=False)

if __name__ == "__main__":
    main()
