import csv
import glob
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

INP_QUERIES = DATA / "llm_api" / "test100" / "inputs" / "test100_with_candidates.jsonl"
LABELS = DATA / "test_labels.json"
OUTDIR = DATA / "llm_api" / "test100" / "metrics"
OUTS_GLOB = str(DATA / "llm_api" / "test100" / "outputs" / "*_test100_fromcand.jsonl")

K_LIST = [1, 3, 5, 10]

def load_qmeta():
    qid2 = {}
    with INP_QUERIES.open("r", encoding="utf-8") as f:
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
        elif isinstance(v, dict) and "gold" in v and isinstance(v["gold"], list):
            lab[qid] = [int(x) for x in v["gold"]]
    return lab

def is_error(val):
    if val is None:
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "none", "null", "nil", "no_error", "ok"):
            return False
    # 其他情况一律认为有错（包括具体异常字符串）
    return bool(val)

def get_pred_list(rec):
    # 优先读 clause_ids（run_api.py 默认写这个）
    for key in ["clause_ids", "pred", "pred_ids", "selected_clause_ids"]:
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
    # 兜底：从 text 里抓数字
    txt = rec.get("text") or ""
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

def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def method_name(p: Path) -> str:
    suf = "_test100_fromcand.jsonl"
    return p.name[:-len(suf)] if p.name.endswith(suf) else p.stem

def build_one(out_jsonl: Path, qmeta, labels):
    acc = {"discipline": defaultdict(lambda: defaultdict(float)),
           "type": defaultdict(lambda: defaultdict(float))}
    cnt = {"discipline": defaultdict(int), "type": defaultdict(int)}
    seen = 0
    missing_label = 0
    missing_meta = 0
    error_cnt = 0

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

            errv = rec.get("error")
            has_err = is_error(errv)
            if has_err:
                error_cnt += 1

            pred = [] if has_err else get_pred_list(rec)

            rowm = {f"hit@{k}": hit_at_k(gold, pred, k) for k in K_LIST}
            rowm["mrr"] = mrr_from_pred(gold, pred)

            lat = safe_float(rec.get("latency_s") or rec.get("latency") or rec.get("latency_mean_s"))
            cost = safe_float(rec.get("cost") or rec.get("cost_usd") or rec.get("cost_mean"))
            if lat is not None:
                rowm["latency_mean_s"] = lat
            if cost is not None:
                rowm["cost_mean"] = cost

            for dim in ("discipline", "type"):
                g = meta.get(dim, "")
                if not g:
                    continue
                cnt[dim][g] += 1
                for k, v in rowm.items():
                    acc[dim][g][k] += float(v)

    out_csv = OUTDIR / f"{method_name(out_jsonl)}_test100_fromcand_group_full.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_by","group","n","hit@1","hit@3","hit@5","hit@10","mrr","latency_mean_s","cost_mean"])
        for dim in ("discipline", "type"):
            for g in sorted(cnt[dim].keys()):
                n = cnt[dim][g]
                a = acc[dim][g]
                def avg(key):
                    return (a.get(key, 0.0) / n) if n else 0.0
                w.writerow([dim, g, n,
                            avg("hit@1"), avg("hit@3"), avg("hit@5"), avg("hit@10"),
                            avg("mrr"), avg("latency_mean_s"), avg("cost_mean")])

    print("[OK] wrote:", out_csv)
    print(f"[INFO] seen={seen} missing_label={missing_label} missing_meta={missing_meta} error_cnt={error_cnt}")

def main():
    assert INP_QUERIES.exists(), f"missing {INP_QUERIES}"
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
