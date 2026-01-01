import json
from pathlib import Path
from typing import Any, Dict, List, Optional

RETRIEVAL_ROOT = Path(__file__).resolve().parents[2]   # .../experiments/retrieval
DATA = RETRIEVAL_ROOT / "data"

TEST_Q   = DATA / "test_queries.jsonl"
DETAILS  = DATA / "test_rerank_details.jsonl"          # ✅ 候选来源：rerank 的 top10
OUT      = DATA / "llm_api" / "test100" / "inputs" / "test100_with_candidates.jsonl"

TOPN = 10  # rerank details 里就是 top10

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def get_qid(obj: Dict[str, Any]) -> Optional[int]:
    for k in ("id", "qid", "query_id"):
        if k in obj and obj[k] is not None:
            try:
                return int(obj[k])
            except Exception:
                pass
    return None

def main():
    assert TEST_Q.exists(), f"missing {TEST_Q}"
    assert DETAILS.exists(), f"missing {DETAILS}"

    qs = read_jsonl(TEST_Q)
    details_rows = read_jsonl(DETAILS)

    # qid -> top10(list[dict])
    top_map: Dict[int, List[Dict[str, Any]]] = {}
    for r in details_rows:
        qid = get_qid(r)
        if qid is None:
            continue
        top10 = r.get("top10")
        if isinstance(top10, list) and top10 and isinstance(top10[0], dict):
            top_map[qid] = top10[:TOPN]

    OUT.parent.mkdir(parents=True, exist_ok=True)

    missing_q = 0
    n_written = 0

    with OUT.open("w", encoding="utf-8") as out_f:
        for q in qs:
            qid = int(q["id"])
            raw = str(q.get("query") or "")
            disc = str(q.get("discipline") or "")
            qtype = str(q.get("type") or "")

            cands = top_map.get(qid)
            if not cands:
                missing_q += 1
                cands = []

            lines = ["QUESTION:", raw, "", "CANDIDATES:"]
            for i, c in enumerate(cands, start=1):
                cid = c.get("clause_id", "")
                clause_no = c.get("clause_no", "")
                src = c.get("source", "")
                text = c.get("text", "")
                lines.append(f"{i}) clause_id={cid} | clause_no={clause_no} | src={src} | text={text}")

            rec = {
                "qid": qid,
                "raw_query": raw,
                "query": "\n".join(lines),
                "discipline": disc,
                "type": qtype,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print("[OK] wrote:", OUT)
    print("[INFO] n_written =", n_written)
    print("[WARN] queries missing candidates =", missing_q)

if __name__ == "__main__":
    main()
