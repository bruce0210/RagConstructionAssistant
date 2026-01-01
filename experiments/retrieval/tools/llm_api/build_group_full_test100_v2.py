import csv, json, glob, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

INP = DATA / "llm_api" / "test100" / "inputs" / "test100_with_candidates.jsonl"
LABELS = DATA / "test_labels.json"
OUTDIR = DATA / "llm_api" / "test100" / "metrics"
OUTS_GLOB = str(DATA / "llm_api" / "test100" / "outputs" / "*_test100_fromcand.jsonl")

K_LIST = [1, 3, 5, 10]

def load_qmeta():
    qid2 = {}
    with INP.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = int(obj["qid"])
            qid2[qid] = {
                "discipline": obj.get("discipline", ""),
                "type": obj.get("type", ""),
            }
    return qid2

def load_labels():
    obj = json.loads(LABELS.read_text(encoding="utf-8"))
    lab = {}
    for k, v in obj.items():
        try:
            qid = int(k)
        except Exception:
            continue
        if isinstance(v, list):
            lab[qid] = [int(x) for x in v]
        elif isinstance(v, dict) and isinstance(v.get("gold"), list):
            lab[qid] = [int(x) for x in v["gold"]]
    return lab

def method_name(p: Path) -> str:
    suf = "_test100_fromcand.jsonl"
    return p.name[:-len(suf)] if p.name.endswith(suf) else p.stem

def read_per_query_csv(prefix: str):
    """
    读取 metrics/*_per_query.csv 获取 latency/cost（这些不在 outputs 里）
    """
    fp = OUTDIR / f"{prefix}_test100_fromcand_per_query.csv"
    m = {}
    if not fp.exists():
        return m, fp
    with fp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            qid_raw = row.get("qid") or row.get("id") or row.get("query_id")
            if qid_raw is None:
                continue
            try:
                qid = int(qid_raw)
            except Exception:
                continue
            lat = row.get("latency_s") or row.get("latency") or ""
            cost = row.get("cost") or row.get("cost_usd") or row.get("cost_mean") or ""
            try:
                lat_v = float(lat) if str(lat).strip() != "" else None
            except Exception:
                lat_v = None
            try:
                cost_v = float(cost) if str(cost).strip() != "" else None
            except Exception:
                cost_v = None
            m[qid] = {"latency_s": lat_v, "cost": cost_v}
    return m, fp

def get_pred_list(rec):
    # 关键：你的 outputs 里是 pred_clause_ids
    for key in ["pred_clause_ids", "clause_ids", "pred", "pred_ids", "selected_clause_ids"]:
        if key in rec and rec[key] is not None:
            v = rec[key]
            if isinstance(v, list):
                out = []
                for x in v:
                    try:
                        out.append(int(x))
                    except Exception:
                        pass
                return out
            if isinstance(v, str):
                return [int(x) for x in re.findall(r"\d+", v)]
    # 兜底：从 raw_text/text 抓数字（不推荐，但防止异常）
    txt = rec.get("raw_text") or rec.get("text") or ""
    if isinstance(txt, str) and txt:
        return [int(x) for x in re.findall(r"\d+", txt)]
    return []

def mrr_from_pred(gold, pred):
    gold_set = set(gold)
    for i, pid in enumerate(pred):
        if pid in gold_set:
            return 1.0 / (i + 1)
    return 0.0

def hit_at_k(gold, pred, k):
    gold_set = set(gold)
    return 1.0 if any(pid in gold_set for pid in pred[:k]) else 0.0

def build_one(out_jsonl: Path, qmeta, labels):
    prefix = method_name(out_jsonl)
    perq, perq_path = read_per_query_csv(prefix)

    # sums + counts
    sums = {"discipline": defaultdict(lambda: defaultdict(float)),
            "type": defaultdict(lambda: defaultdict(float))}
    ncnt = {"discipline": defaultdict(int), "type": defaultdict(int)}
    lat_sum = {"discipline": defaultdict(float), "type": defaultdict(float)}
    lat_cnt = {"discipline": defaultdict(int), "type": defaultdict(int)}
    cost_sum = {"discipline": defaultdict(float), "type": defaultdict(float)}
    cost_cnt = {"discipline": defaultdict(int), "type": defaultdict(int)}

    seen = 0
    missing_label = 0
    missing_meta = 0
    empty_pred = 0

    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            seen += 1

            qid_raw = rec.get("qid") or rec.get("id") or rec.get("query_id")
            try:
                qid = int(qid_raw)
            except Exception:
                continue

            gold = labels.get(qid)
            if not gold:
                missing_label += 1
                continue

            meta = qmeta.get(qid)
            if not meta:
                missing_meta += 1
                continue

            pred = get_pred_list(rec)
            if not pred:
                empty_pred += 1

            rowm = {f"hit@{k}": hit_at_k(gold, pred, k) for k in K_LIST}
            rowm["mrr"] = mrr_from_pred(gold, pred)

            # latency/cost from per_query.csv
            lat = perq.get(qid, {}).get("latency_s")
            cost = perq.get(qid, {}).get("cost")

            for dim in ("discipline", "type"):
                g = meta.get(dim, "")
                if not g:
                    continue
                ncnt[dim][g] += 1
                for k, v in rowm.items():
                    sums[dim][g][k] += float(v)
                if lat is not None:
                    lat_sum[dim][g] += float(lat)
                    lat_cnt[dim][g] += 1
                if cost is not None:
                    cost_sum[dim][g] += float(cost)
                    cost_cnt[dim][g] += 1

    out_csv = OUTDIR / f"{prefix}_test100_fromcand_group_full.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_by","group","n","hit@1","hit@3","hit@5","hit@10","mrr","latency_mean_s","cost_mean"])
        for dim in ("discipline", "type"):
            for g in sorted(ncnt[dim].keys()):
                n = ncnt[dim][g]
                def avg_metric(key):
                    return (sums[dim][g].get(key, 0.0) / n) if n else 0.0
                lat_mean = (lat_sum[dim][g] / lat_cnt[dim][g]) if lat_cnt[dim][g] else 0.0
                cost_mean = (cost_sum[dim][g] / cost_cnt[dim][g]) if cost_cnt[dim][g] else 0.0
                w.writerow([dim, g, n,
                            avg_metric("hit@1"), avg_metric("hit@3"), avg_metric("hit@5"), avg_metric("hit@10"),
                            avg_metric("mrr"), lat_mean, cost_mean])

    print("[OK] wrote:", out_csv)
    print(f"[INFO] prefix={prefix}")
    print(f"[INFO] seen={seen} missing_label={missing_label} missing_meta={missing_meta} empty_pred={empty_pred}")
    print(f"[INFO] per_query_csv={perq_path} exists={perq_path.exists()} rows={len(perq)}")

def main():
    assert INP.exists(), f"missing {INP}"
    assert LABELS.exists(), f"missing {LABELS}"
    qmeta = load_qmeta()
    labels = load_labels()
    outs = sorted(glob.glob(OUTS_GLOB))
    if not outs:
        raise RuntimeError(f"No outputs found: {OUTS_GLOB}")
    for fp in outs:
        build_one(Path(fp), qmeta, labels)

if __name__ == "__main__":
    main()
