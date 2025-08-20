# core/retrieval/ingest_docx.py
from __future__ import annotations
from pathlib import Path
import os, re, json
from typing import List, Dict, Any

from docx import Document
from lxml import etree
import numpy as np
from sentence_transformers import SentenceTransformer


# 统一的命名空间表和工具函数 放在 import 后
XML_NS = {
    "w":  "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a":  "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r":  "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "pic":"http://schemas.openxmlformats.org/drawingml/2006/picture",
}

def _ns(elem=None):
    ns = dict(XML_NS)
    if elem is not None and getattr(elem, "nsmap", None):
        # 合并当前节点可见的 ns（过滤 None 前缀）
        ns.update({k:v for k,v in elem.nsmap.items() if k})
    return ns

# ---------- Paths (exported) ----------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR:   Path = REPO_ROOT / "data"
INDEX_DIR:  Path = DATA_DIR / "index"
MEDIA_DIR:  Path = DATA_DIR / "media"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "REPO_ROOT", "DATA_DIR", "INDEX_DIR", "MEDIA_DIR",
    "parse_docx_into_clauses", "get_embedder", "embed_texts", "build_faiss_index",
    "write_faiss_index",
]

# ---------- DOCX parsing ----------
# 支持 X、X.X、X.X.X
CLAUSE_PAT = re.compile(r"^\s*((?:\d+\.){0,2}\d+)\b")
EXPLAIN_HEAD = re.compile(r"^\s*条文说明[:：]?\s*$")

def _slugify(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)

def _para_lines(p_elem) -> List[str]:
    """
    把一个 <w:p> 段落按手动换行 <w:br/> 切成多“行”。
    ↵=新条；¶=不切分（自动换行不会出现在 XML）。
    """
    ns = p_elem.nsmap or {}
    xp_text = etree.XPath(".//w:t", namespaces=ns)  # 预编译 XPath，避免 namespaces= 报错
    lines, buf = [], []
    for child in p_elem.iterchildren():
        tag = etree.QName(child.tag).localname if hasattr(child, "tag") else ""
        if tag == "r":
            for t in xp_text(child):
                if t.text:
                    buf.append(t.text)
        elif tag == "br":  # ↵
            s = "".join(buf).strip()
            if s: lines.append(s)
            buf = []
        else:
            pass
    tail = "".join(buf).strip()
    if tail:
        lines.append(tail)
    return lines

def _iter_body_elems(doc: Document):
    """
    遍历 body：
      - <w:p> 段落 → 按 <w:br/> 切为多“行”：yield ("line", p_elem, line_text)
      - <w:tbl> 表格 → 原样：yield ("tbl", tbl_elem, "")
    """
    body = doc.element.body
    for child in body.iterchildren():
        tag = etree.QName(child.tag).localname
        if tag == "p":
            # 关键：按手动换行 <w:br/> 切成多“行”
            for line in _para_lines(child):
                if line.strip():
                    yield ("line", child, line.strip())
        elif tag == "tbl":
            yield ("tbl", child, "")

def _extract_images_from_elem(doc: Document, elem, save_dir: Path) -> List[str]:
    ns = _ns(elem)
    xp_blip = etree.XPath(".//a:blip", namespaces=ns)

    blips = xp_blip(elem)
    rels = doc.part._rels
    saved: List[str] = []
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, blip in enumerate(blips):
        rId = (blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
               or blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}link"))
        if not rId or rId not in rels:
            continue

        rel = rels[rId]
        # 外链图片没有 target_part，直接跳过
        if getattr(rel, "is_external", False):
            continue

        part = rel.target_part  # ✅ 正确属性名
        # part.partname 可能是 PackURI，转成字符串再取扩展名
        partname = str(part.partname)
        ext = os.path.splitext(os.path.basename(partname))[-1] or ".png"

        out_path = save_dir / f"img_{i}{ext}"
        with open(out_path, "wb") as f:
            f.write(part.blob)
        saved.append(str(out_path))

    return saved

def parse_docx_into_clauses(docx_path: Path) -> List[Dict[str, Any]]:
    """
    切条规则：
      1) 遇到 “X / X.X / X.X.X” 编号 → 开始新的 base_no。
      2) 同一段落里的 ↵（<w:br/>）→ 认为是“下一条规范”，在当前 base_no 下生成子条款。
      3) 段落 ¶（<w:p> 结束）不切分，只把文本并入当前条款。
    生成的条款号形式：base_no-1, base_no-2, ...（如需只保留 base_no，可把 cur_no 改为 base_no）
    """
    doc = Document(str(docx_path))
    doc_slug = _slugify(Path(docx_path).stem)
    media_root = MEDIA_DIR / doc_slug

    clauses: List[Dict[str, Any]] = []
    cur_text_parts: List[str] = []
    cur_media: List[str] = []
    base_no: str | None = None
    sub_idx: int = 0
    cur_no: str | None = None
    in_explain = False

    def flush():
        nonlocal cur_no, cur_text_parts, cur_media, in_explain
        if cur_no and cur_text_parts:
            clauses.append({
                "source": Path(docx_path).name,
                "clause_no": cur_no,
                "text": "\n".join(cur_text_parts).strip(),
                "media": cur_media[:],
            })
        cur_no = None
        cur_text_parts = []
        cur_media = []
        in_explain = False

    last_line_elem = None      # 识别同一段 p 内的第几行（是否由 ↵ 切出）
    line_seq_in_elem = 0
    last_img_elem = None       # 防止同一段图片重复提取

    for kind, elem, text in _iter_body_elems(doc):
        # 每个 elem 的图片只抽一次，归到“当前条款”
        if cur_no and elem is not last_img_elem:
            save_dir = media_root / f"clause_{len(clauses)}"
            cur_media += _extract_images_from_elem(doc, elem, save_dir)
            last_img_elem = elem

        if kind == "line":
            # 是否同一 <w:p> 内的后续“行”（由 ↵ 切出）
            if elem is last_line_elem:
                line_seq_in_elem += 1
            else:
                last_line_elem = elem
                line_seq_in_elem = 0

            m = CLAUSE_PAT.match(text)
            if m:
                # 新编号 → 开新 base_no
                flush()
                base_no = m.group(1)
                sub_idx = 1
                cur_no = f"{base_no}-{sub_idx}"
                cur_text_parts.append(text)
                continue

            if EXPLAIN_HEAD.match(text):
                in_explain = True
                cur_text_parts.append("条文说明:")
                continue

            if base_no and cur_no:
                # 同一段内的第二/三…“行”= ↵ → 切到下一条
                if line_seq_in_elem > 0:
                    flush()
                    sub_idx += 1
                    cur_no = f"{base_no}-{sub_idx}"
                if text:
                    cur_text_parts.append(text)

        elif kind == "tbl":
            # 表格文本忽略（通常以图片形式保存）
            pass

    flush()
    return [c for c in clauses if len(c.get("text", "")) >= 3]

# ---------- Embedding / Index ----------

def get_embedder(model_name: str) -> SentenceTransformer:
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
    return SentenceTransformer(model_name, device=device)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def build_faiss_index(vecs: np.ndarray, use_gpu_if_possible: bool = True):
    import faiss  # 局部导入，避免 UI 导入模块阶段出错
    dim = int(vecs.shape[1])
    index = faiss.IndexFlatIP(dim)
    if use_gpu_if_possible:
        try:
            if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)  # 显式用第0卡
        except Exception as e:
            print(f"[faiss] GPU fallback to CPU due to: {e}")
    index.add(vecs.astype("float32"))
    return index


def write_faiss_index(index, path: Path):
    """把索引写到磁盘。若是 GPU 索引，先迁回 CPU 再写。"""
    import faiss
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 如果是 GPU 索引会成功返回 CPU 索引；CPU 索引会抛异常，直接沿用原索引即可
    try:
        index = faiss.index_gpu_to_cpu(index)
    except Exception:
        pass

    faiss.write_index(index, str(path))


# ---------- CLI (optional) ----------

def _cli_ingest_docx(docx_file: str, model_name: str = "BAAI/bge-small-zh-v1.5"):
    p = Path(docx_file)
    assert p.exists(), f"找不到文件：{p}"
    clauses = parse_docx_into_clauses(p)
    print(f"[ingest] 条款：{len(clauses)}")

    model = get_embedder(model_name)
    texts = [c["text"] for c in clauses]
    vecs = embed_texts(model, texts, batch_size=64)
    index = build_faiss_index(vecs, use_gpu_if_possible=True)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    write_faiss_index(index, INDEX_DIR / "faiss.index")
    with open(INDEX_DIR / "meta.jsonl", "w", encoding="utf-8") as f:
        for c in clauses:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[ingest] 索引写入：{INDEX_DIR / 'faiss.index'}")
    print(f"[ingest] 元数据写入：{INDEX_DIR / 'meta.jsonl'}")
    print(f"[ingest] 图片目录：{MEDIA_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", help="docx 规范路径")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    args = ap.parse_args()
    _cli_ingest_docx(args.docx, model_name=args.model)
