import argparse, json, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    for fn in os.listdir(args.metrics_dir):
        if fn.endswith("_overall.json"):
            p = os.path.join(args.metrics_dir, fn)
            obj = json.load(open(p,"r",encoding="utf-8"))
            rows.append({
                "provider": obj.get("provider"),
                "model": obj.get("model"),
                "n": obj.get("n"),
                "hit@1": obj.get("hit@1"),
                "hit@3": obj.get("hit@3"),
                "hit@10": obj.get("hit@10"),
                "mrr": obj.get("mrr"),
                "lat_mean_s": obj.get("latency_mean_s"),
                "lat_p95_s": obj.get("latency_p95_s"),
                "tokens_sum": obj.get("total_tokens_sum"),
                "cost_sum": obj.get("cost_sum"),
                "cost_mean": obj.get("cost_mean"),
                "cost_unit": obj.get("cost_unit"),
            })
    df = pd.DataFrame(rows).sort_values(["provider","model"])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
