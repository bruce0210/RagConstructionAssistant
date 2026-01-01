import csv
import glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

METDIR = DATA / "llm_api" / "test100" / "metrics"

# 输入：我们刚生成的 group_full
GLOB_IN = str(METDIR / "*_test100_fromcand_group_full.csv")

# 输出：和 baseline 一样的 long / wide（ALIGNED_NO_NAN）
OUT_TYPE_LONG = DATA / "llm_api_by_type_long_TEST100_FULL_ALIGNED_NO_NAN.csv"
OUT_TYPE_WIDE = DATA / "llm_api_by_type_wide_TEST100_FULL_ALIGNED_NO_NAN.csv"
OUT_DISC_LONG = DATA / "llm_api_by_discipline_long_TEST100_FULL_ALIGNED_NO_NAN.csv"
OUT_DISC_WIDE = DATA / "llm_api_by_discipline_wide_TEST100_FULL_ALIGNED_NO_NAN.csv"

# 你要对齐的指标（LLM 没有 ndcg@10，所以这里用 hit@k + mrr + latency/cost）
METRICS = ["n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "latency_mean_s", "cost_mean"]

def method_name(p: Path) -> str:
    suf = "_test100_fromcand_group_full.csv"
    return p.name[:-len(suf)] if p.name.endswith(suf) else p.stem

def read_one(fp: Path):
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def build_tables(group_dim: str, out_long: Path, out_wide: Path):
    files = sorted(glob.glob(GLOB_IN))
    if not files:
        raise RuntimeError(f"No inputs found: {GLOB_IN}")

    # method -> group -> metric -> value
    data = defaultdict(lambda: defaultdict(dict))
    groups_by_method = {}

    for fpath in files:
        fp = Path(fpath)
        m = method_name(fp)
        rows = read_one(fp)

        groups = set()
        for row in rows:
            if row.get("group_by") != group_dim:
                continue
            g = row.get("group")
            if not g:
                continue
            groups.add(g)
            for k in METRICS:
                v = row.get(k, "")
                if v == "" or v is None:
                    continue
                try:
                    # n 是 int，其余 float
                    if k == "n":
                        data[m][g][k] = int(float(v))
                    else:
                        data[m][g][k] = float(v)
                except Exception:
                    pass

        groups_by_method[m] = groups

    methods = sorted(groups_by_method.keys())
    common_groups = set.intersection(*(groups_by_method[m] for m in methods)) if methods else set()
    common_groups = sorted(common_groups)

    print(f"[INFO] dim={group_dim} methods={methods}")
    print(f"[INFO] common_groups({group_dim}) = {common_groups}")

    # 输出 long：method, group_dim, metric, value
    out_long.parent.mkdir(parents=True, exist_ok=True)
    with out_long.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", group_dim, "metric", "value"])
        for m in methods:
            for g in common_groups:
                for k in METRICS:
                    if k not in data[m][g]:
                        continue
                    w.writerow([m, g, k, data[m][g][k]])

    # 输出 wide：每行一个 method，列是 group|metric
    wide_cols = ["method"]
    for g in common_groups:
        for k in METRICS:
            wide_cols.append(f"{g}|{k}")

    with out_wide.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(wide_cols)
        for m in methods:
            row = [m]
            for g in common_groups:
                for k in METRICS:
                    row.append(data[m][g].get(k, ""))
            w.writerow(row)

    print("[OK] wrote:", out_long)
    print("[OK] wrote:", out_wide)

def main():
    build_tables("type", OUT_TYPE_LONG, OUT_TYPE_WIDE)
    build_tables("discipline", OUT_DISC_LONG, OUT_DISC_WIDE)

if __name__ == "__main__":
    main()
