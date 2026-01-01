import os, json, argparse
from tqdm import tqdm

from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig

def pick(obj, keys, default=None):
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return default

def render_passage(obj):
    # 尽量兼容你 meta.jsonl 的字段命名（不强依赖某一个 key）
    text = pick(obj, ["text", "content", "chunk", "clause_text", "passage"], "")
    heading = pick(obj, ["heading", "title", "section_title"], "")
    scope = pick(obj, ["scope", "parent_scope"], "")
    parts = [p for p in [heading, scope, text] if p]
    return "\n".join(parts) if parts else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="jinaai/jina-colbert-v2")
    ap.add_argument("--meta", default="/home/RagConstructionAssistant/data/index/meta.jsonl")
    ap.add_argument("--out_root", default="/home/RagConstructionAssistant/data/index/colbert")
    ap.add_argument("--index_name", default="colbert_jina_v2")
    ap.add_argument("--limit", type=int, default=0, help=">0 用于 smoke test（先小规模跑通）")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    pid2clause_path = os.path.join(args.out_root, f"{args.index_name}_pid2clause.json")

    passages = []
    pid2clause = []

    with open(args.meta, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading meta")):
            if args.limit and i >= args.limit:
                break
            obj = json.loads(line)
            clause_id = pick(obj, ["clause_id", "id", "clauseId", "clauseID"], None)
            if clause_id is None:
                raise KeyError("meta.jsonl 中找不到 clause_id / id 字段，请检查字段名")
            clause_id = int(clause_id)

            passage = render_passage(obj)
            if not passage:
                # 最保守：至少别空
                passage = json.dumps(obj, ensure_ascii=False)

            passages.append(passage)
            pid2clause.append(clause_id)

    with open(pid2clause_path, "w", encoding="utf-8") as w:
        json.dump(pid2clause, w, ensure_ascii=False)

    cfg = ColBERTConfig(
        root=args.out_root,
        nbits=2,          # ColBERTv2 常用压缩配置
        doc_maxlen=256,
        query_maxlen=64,
    )

    # nranks=1 表示单卡；你是 L20 一张卡就够
    with Run().context(RunConfig(nranks=1)):
        indexer = Indexer(checkpoint=args.checkpoint, config=cfg)
        indexer.index(name=args.index_name, collection=passages, overwrite=True)

    print(f"[OK] wrote: {pid2clause_path}")
    print(f"[OK] built index: root={args.out_root}, name={args.index_name}")

if __name__ == "__main__":
    main()
