# annotate_dev.py
import json
from pathlib import Path

from search import search  # 复用确认无误的 search()

ROOT = Path(__file__).resolve().parent
DEV_QUERIES = ROOT / "data" / "dev_queries.jsonl"
DEV_LABELS = ROOT / "data" / "dev_labels.json"

def load_queries():
    queries = []
    with DEV_QUERIES.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries

def main():
    queries = load_queries()
    labels = {}

    print("共有 {} 个查询需要标注 gold 条文。".format(len(queries)))
    print("你可以在每个问题上输入：")
    print("  - 主要条文 id（必填，可逗号分隔，如 123,456）")
    print("  - 回车跳过（暂时不标）")
    print("-" * 80)

    for q in queries:
        qid = q["id"]
        qtext = q["query"]
        print("\n" + "=" * 80)
        print(f"[Q{qid}] {qtext}")

        # 调用检索
        results = search(qtext, top_k=10)
        if not results:
            print("  没有检索到结果，跳过。")
            continue

        for r in results:
            print(f"\n  Top {r['rank']}  score={r['score']*100:.2f}%")
            print(f"    id: {r['id']}")
            print(f"    来源: {r['source']}")
            print(f"    条号: {r['clause_no']} / {r['clause']}")
            txt = (r['text'] or "").replace("\n", " ")
            print(f"    内容: {txt[:160]}...")

        inp = input("\n请输入本题的主要条文 id（可逗号分隔，直接回车=暂不标注）：").strip()
        if not inp:
            continue

        try:
            gold_ids = [int(x) for x in inp.split(",") if x.strip()]
        except ValueError:
            print("输入格式错误，跳过本题。")
            continue

        labels[str(qid)] = gold_ids
        print(f"已记录 Q{qid} 的 gold 条文 id: {gold_ids}")

    # 写入文件
    DEV_LABELS.parent.mkdir(parents=True, exist_ok=True)
    with DEV_LABELS.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("\n标注完成，已保存到", DEV_LABELS)

if __name__ == "__main__":
    main()
