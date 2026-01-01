import ast
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

METHODS = ["bm25", "dense", "dpr", "hybrid", "rerank", "colbert"]
METRIC_ORDER = ["hit@1", "hit@10", "hit@3", "n", "ndcg@10"]

# 维度名兼容（避免不同脚本写成 qtype / question_type）
DIM_CANON = {
    "type": {"type", "qtype", "question_type", "questiontype"},
    "discipline": {"discipline", "domain", "field"},
    "all": {"all"},
}

TUPLE_RE = re.compile(r"""\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)""")

def canonical_dim(dim_raw: str) -> str:
    d = (dim_raw or "").strip().lower()
    for canon, aliases in DIM_CANON.items():
        if d in aliases:
            return canon
    return d  # fallback

def parse_tuple_key(k: str):
    # 先试 literal_eval
    try:
        t = ast.literal_eval(k)
        if isinstance(t, tuple) and len(t) == 2:
            return str(t[0]), str(t[1])
    except Exception:
        pass
    # 再试正则
    m = TUPLE_RE.search(k)
    if m:
        return m.group(1), m.group(2)
    return None

def load_group_metrics(method: str) -> dict:
    p = DATA_DIR / f"test_{method}_group_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))

def build_tables(group_dim: str, out_prefix: str):
    per_method = {}
    dims_found = {}

    for m in METHODS:
        obj = load_group_metrics(m)
        groups = {}
        dims = set()

        for k, v in obj.items():
            parsed = parse_tuple_key(k)
            if not parsed:
                continue
            dim_raw, name = parsed
            dim = canonical_dim(dim_raw)
            dims.add(dim)
            if dim != group_dim:
                continue
            if isinstance(v, dict):
                groups[name] = v

        per_method[m] = groups
        dims_found[m] = dims

    # 打印每个方法识别到的 group 数量（帮助定位问题，输出很短）
    print(f"[DEBUG] dim={group_dim} group counts:")
    for m in METHODS:
        print(f"  - {m}: {len(per_method[m])} groups (dims_found={sorted(list(dims_found[m]))})")

    common_groups = None
    for m in METHODS:
        s = set(per_method[m].keys())
        common_groups = s if common_groups is None else (common_groups & s)

    common_groups = sorted(list(common_groups or []))
    if not common_groups:
        raise RuntimeError(
            f"No common groups found for dim={group_dim}. "
            f"At least one method has 0 groups for this dim. "
            f"See [DEBUG] lines above."
        )

    # long
    long_path = DATA_DIR / f"{out_prefix}_long_TEST100_ALIGNED_NO_NAN.csv"
    with long_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", group_dim, "metric", "value"])
        for m in METHODS:
            for g in common_groups:
                metrics = per_method[m][g]
                for met in METRIC_ORDER:
                    if met in metrics:
                        w.writerow([m, g, met, metrics[met]])

    # wide
    wide_path = DATA_DIR / f"{out_prefix}_wide_TEST100_ALIGNED_NO_NAN.csv"
    header = ["method"] + [f"{g}|{met}" for g in common_groups for met in METRIC_ORDER]
    with wide_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in METHODS:
            row = [m]
            for g in common_groups:
                metrics = per_method[m][g]
                for met in METRIC_ORDER:
                    row.append(metrics.get(met, ""))
            w.writerow(row)

    print("[OK] wrote:", long_path)
    print("[OK] wrote:", wide_path)
    print("[INFO] common_groups:", common_groups)

def main():
    build_tables("type", out_prefix="baseline_by_type")
    build_tables("discipline", out_prefix="baseline_by_discipline")

if __name__ == "__main__":
    main()
