import csv
import glob
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]          # .../experiments/retrieval
METDIR = ROOT / "data" / "llm_api" / "test100" / "metrics"
OUTDIR = ROOT / "data"

# 输出文件（放在 data/ 根目录，和 baseline_by_* 同级，便于你后续统一画图）
OUT_TYPE_LONG = OUTDIR / "llm_api_by_type_long_TEST100_ALIGNED_NO_NAN.csv"
OUT_TYPE_WIDE = OUTDIR / "llm_api_by_type_wide_TEST100_ALIGNED_NO_NAN.csv"
OUT_DISC_LONG = OUTDIR / "llm_api_by_discipline_long_TEST100_ALIGNED_NO_NAN.csv"
OUT_DISC_WIDE = OUTDIR / "llm_api_by_discipline_wide_TEST100_ALIGNED_NO_NAN.csv"

METRICS = ["hit@1", "mrr", "latency_mean_s", "cost_mean"]

def method_name_from_path(p: str) -> str:
    # e.g. openai_gpt-4o_test100_fromcand_group.csv -> openai_gpt-4o
    base = os.path.basename(p)
    suf = "_test100_fromcand_group.csv"
    if base.endswith(suf):
        return base[:-len(suf)]
    return base.replace(".csv", "")

def read_group_csv(path: str):
    # return: dim -> group -> {metric: value, n: int}
    dim_map = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dim = row["group_by"].strip()          # discipline / type
            grp = row["group"].strip()
            d = {"n": int(float(row["n"]))}
            for m in METRICS:
                d[m] = float(row[m])
            dim_map[dim][grp] = d
    return dim_map

def build_tables(dim_key: str, dim_col_name: str, out_long: Path, out_wide: Path):
    files = sorted(glob.glob(str(METDIR / "*_test100_fromcand_group.csv")))
    if not files:
        raise RuntimeError(f"No group.csv found under {METDIR}")

    per_method = {}
    for fp in files:
        name = method_name_from_path(fp)
        dm = read_group_csv(fp)
        per_method[name] = dm.get(dim_key, {})

    # 对齐：取所有方法的交集组
    group_sets = [set(gmap.keys()) for gmap in per_method.values()]
    common = set.intersection(*group_sets) if group_sets else set()
    common = sorted(common)

    if not common:
        # 打印 debug 方便定位
        print(f"[DEBUG] dim={dim_key} group counts:")
        for m, gmap in per_method.items():
            print(f"  - {m}: {len(gmap)} groups")
        raise RuntimeError(f"No common groups found for dim={dim_key}")

    # long
    with open(out_long, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", dim_col_name, "metric", "value"])
        for method in sorted(per_method.keys()):
            for grp in common:
                for metric in ["n"] + METRICS:
                    val = per_method[method][grp]["n"] if metric == "n" else per_method[method][grp][metric]
                    w.writerow([method, grp, metric, val])

    # wide: method as row, columns are "{group}|{metric}"
    cols = []
    for grp in common:
        for metric in ["n"] + METRICS:
            cols.append(f"{grp}|{metric}")

    with open(out_wide, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols)
        for method in sorted(per_method.keys()):
            row = [method]
            for grp in common:
                row.append(per_method[method][grp]["n"])
                for metric in METRICS:
                    row.append(per_method[method][grp][metric])
            w.writerow(row)

    print(f"[OK] wrote: {out_long}")
    print(f"[OK] wrote: {out_wide}")
    print(f"[INFO] common_groups({dim_key}) = {common}")

def main():
    build_tables("type", "type", OUT_TYPE_LONG, OUT_TYPE_WIDE)
    build_tables("discipline", "discipline", OUT_DISC_LONG, OUT_DISC_WIDE)

if __name__ == "__main__":
    main()
