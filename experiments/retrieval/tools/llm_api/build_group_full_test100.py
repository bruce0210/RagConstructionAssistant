import csv
import glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]  # .../experiments/retrieval
METDIR = ROOT / "data" / "llm_api" / "test100" / "metrics"
INP = ROOT / "data" / "llm_api" / "test100" / "inputs" / "test100_with_candidates.jsonl"

# 读取 qid -> discipline/type
import json
qid2meta = {}
with INP.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = int(obj["qid"])
        qid2meta[qid] = {
            "discipline": obj.get("discipline", ""),
            "type": obj.get("type", ""),
        }

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def agg_write(per_query_csv: Path, out_csv: Path):
    with per_query_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        qid_col = pick_col(cols, ["qid", "id", "query_id"])
        hit1 = pick_col(cols, ["hit@1", "hit1"])
        hit3 = pick_col(cols, ["hit@3", "hit3"])
        hit5 = pick_col(cols, ["hit@5", "hit5"])
        hit10 = pick_col(cols, ["hit@10", "hit10"])
        mrr = pick_col(cols, ["mrr", "mrr@10", "mrr@k"])
        lat = pick_col(cols, ["latency_s", "latency", "latency_mean_s"])
        cost = pick_col(cols, ["cost", "cost_usd", "cost_mean"])

        need = [qid_col, hit1, hit3, hit5, hit10, mrr]
        if any(x is None for x in need):
            raise RuntimeError(
                f"Missing required columns in {per_query_csv.name}. "
                f"Have={cols}. Need qid/hit@1/hit@3/hit@5/hit@10/mrr."
            )

        # group_by -> group -> sums
        acc = {
            "discipline": defaultdict(lambda: defaultdict(float)),
            "type": defaultdict(lambda: defaultdict(float)),
        }
        cnt = {
            "discipline": defaultdict(int),
            "type": defaultdict(int),
        }

        for row in r:
            qid = int(float(row[qid_col]))
            meta = qid2meta.get(qid)
            if not meta:
                continue

            vals = {
                "hit@1": float(row[hit1]),
                "hit@3": float(row[hit3]),
                "hit@5": float(row[hit5]),
                "hit@10": float(row[hit10]),
                "mrr": float(row[mrr]),
            }
            if lat is not None and row.get(lat) not in (None, ""):
                vals["latency_mean_s"] = float(row[lat])
            if cost is not None and row.get(cost) not in (None, ""):
                vals["cost_mean"] = float(row[cost])

            for dim in ("discipline", "type"):
                g = meta.get(dim, "")
                if not g:
                    continue
                cnt[dim][g] += 1
                for k, v in vals.items():
                    acc[dim][g][k] += v

    # 写 full group
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_by","group","n","hit@1","hit@3","hit@5","hit@10","mrr","latency_mean_s","cost_mean"])
        for dim in ("discipline", "type"):
            for g in sorted(cnt[dim].keys()):
                n = cnt[dim][g]
                a = acc[dim][g]
                def avg(key):
                    return (a.get(key, 0.0) / n) if n else 0.0
                w.writerow([
                    dim, g, n,
                    avg("hit@1"), avg("hit@3"), avg("hit@5"), avg("hit@10"),
                    avg("mrr"),
                    avg("latency_mean_s"),
                    avg("cost_mean"),
                ])

    print("[OK] wrote:", out_csv)

def main():
    files = sorted(glob.glob(str(METDIR / "*_test100_fromcand_per_query.csv")))
    if not files:
        raise RuntimeError(f"No per_query csv under {METDIR}")

    for fp in files:
        fp = Path(fp)
        out = fp.with_name(fp.name.replace("_per_query.csv", "_group_full.csv"))
        agg_write(fp, out)

if __name__ == "__main__":
    main()
