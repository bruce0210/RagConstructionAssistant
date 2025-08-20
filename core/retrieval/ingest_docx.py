# core/retrieval/ingest_docx.py
from __future__ import annotations
from pathlib import Path
import os, re, json, hashlib
from typing import List, Dict, Any

from docx import Document
from lxml import etree
import numpy as np
from sentence_transformers import SentenceTransformer

XML_NS = {
    "w":  "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a":  "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r":  "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/wordprocessingDrawing",
    "pic":"http://schemas.openxmlformats.org/drawingml/2006/picture",
}
def _ns(elem=None):
    ns = dict(XML_NS)
    if elem is not None and getattr(elem, "nsmap", None):
        ns.update({k:v for k,v in elem.nsmap.items() if k})
    return ns

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR:   Path = REPO_ROOT / "data"
INDEX_DIR:  Path = DATA_DIR / "index"
MEDIA_DIR:  Path = DATA_DIR / "media"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "REPO_ROOT", "DATA_DIR", "INDEX_DIR", "MEDIA_DIR",
    "parse_docx_into_clauses", "get_embedder", "embed_texts",
    "build_faiss_index", "write_faiss_index",
    "load_faiss_index_if_exists", "count_existing_meta_lines",
    "build_or_append_faiss_index", "write_meta_jsonl",
    "file_blake2b_hex", "load_seen_doc_hashes",
]

# 识别「5.0.9」「11.0.2」等编号；实际条款号写成 5.0.9-1、5.0.9-2 …
CLAUSE_PAT = re.compile(r"^\s*((?:\d+\.){0,2}\d+)\b")
EXPLAIN_HEAD = re.compile(r"^\s*条文说明[:：]?\s*$")

def _slugify(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)

def _para_lines(p_elem) -> List[str]:
    ns = p_elem.nsmap or {}
    xp_text = etree.XPath(".//w:t", namespaces=ns)
    lines, buf = [], []
    for child in p_elem.iterchildren():
        tag = etree.QName(child.tag).localname if hasattr(child, "tag") else ""
        if tag == "r":
            for t in xp_text(child):
                if t.text:
                    buf.append(t.text)
        elif tag == "br":
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
    """顺序遍历正文的 p / tbl，p 内按换行(br)切分为多行"""
    body = doc.element.body
    for child in body.iterchildren():
        tag = etree.QName(child.tag).localname
        if tag == "p":
            for line in _para_lines(child):
                if line.strip():
                    yield ("line", child, line.strip())
        elif tag == "tbl":
            yield ("tbl", child, "")

def _extract_images_from_elem(doc: Document, elem, save_dir: Path) -> List[str]:
    """抓取当前段落/表格里的所有图片，保存到 save_dir，返回本地文件路径列表"""
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
        if getattr(rel, "is_external", False):
            continue
        part = rel.target_part
        partname = str(part.partname)
        ext = os.path.splitext(os.path.basename(partname))[-1] or ".png"
        out_path = save_dir / f"img_{i}{ext}"
        with open(out_path, "wb") as f:
            f.write(part.blob)
        saved.append(str(out_path))
    return saved

def parse_docx_into_clauses(docx_path: Path) -> List[Dict[str, Any]]:
    """
    解析 docx 为条款列表。
    为每条记录补充：
      - clause_no（如 5.0.9-1）与兼容字段 clause
      - clause_idx（从 1 开始的稳定顺序号）
      - media_key（'clause_{clause_idx}'，供前端映射 OSS 目录）
      - media（本地抽到的多张图片路径列表；前端会用 media_key 覆盖为 OSS 图）
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
            clause_idx = len(clauses) + 1  # 1-based，对应 OSS 子目录
            clauses.append({
                "source": Path(docx_path).name,
                "clause_no": cur_no,
                "clause": cur_no,
                "clause_idx": clause_idx,
                "media_key": f"clause_{clause_idx}",
                "text": "\n".join(cur_text_parts).strip(),
                "media": cur_media[:],        # 允许多图
            })
        cur_no = None
        cur_text_parts = []
        cur_media = []
        in_explain = False

    last_line_elem = None
    line_seq_in_elem = 0
    last_img_elem = None  # 确保每个 elem 只抽一次图

    for kind, elem, text in _iter_body_elems(doc):
        if kind == "line":
            if elem is last_line_elem:
                line_seq_in_elem += 1
            else:
                last_line_elem = elem
                line_seq_in_elem = 0

            m = CLAUSE_PAT.match(text)
            if m:
                flush()
                base_no = m.group(1)
                sub_idx = 1
                cur_no = f"{base_no}-{sub_idx}"
                cur_text_parts.append(text)
            elif EXPLAIN_HEAD.match(text):
                in_explain = True
                cur_text_parts.append("条文说明:")
            else:
                if base_no and cur_no:
                    if line_seq_in_elem > 0:
                        flush()
                        sub_idx += 1
                        cur_no = f"{base_no}-{sub_idx}"
                    if text:
                        cur_text_parts.append(text)

        elif kind == "tbl":
            pass

        # 决定好 cur_no 之后，再抽图，避免挂到上一条
        if cur_no and elem is not last_img_elem:
            save_dir = media_root / f"clause_{len(clauses)+1}"
            cur_media += _extract_images_from_elem(doc, elem, save_dir)
            last_img_elem = elem

    # 文档收尾
    flush()
    return [c for c in clauses if len(c.get("text", "")) >= 3]


def get_embedder(model_name: str) -> SentenceTransformer:
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
    return SentenceTransformer(model_name, device=device)

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

def build_faiss_index(vecs: np.ndarray, use_gpu_if_possible: bool = True):
    import faiss
    dim = int(vecs.shape[1])
    index = faiss.IndexFlatIP(dim)
    if use_gpu_if_possible:
        try:
            if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"[faiss] GPU fallback to CPU due to: {e}")
    index.add(vecs.astype("float32"))
    return index

def write_faiss_index(index, path: Path):
    import faiss
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        index = faiss.index_gpu_to_cpu(index)
    except Exception:
        pass
    faiss.write_index(index, str(path))

def load_faiss_index_if_exists(path: Path):
    import faiss
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return faiss.read_index(str(p))
    return None

def count_existing_meta_lines(path: Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def build_or_append_faiss_index(new_vecs: np.ndarray, index_path: Path, use_gpu_if_possible: bool = True):
    import faiss
    dim = int(new_vecs.shape[1])
    cpu_index = load_faiss_index_if_exists(index_path)
    if cpu_index is None:
        cpu_index = faiss.IndexFlatIP(dim)
    else:
        if cpu_index.d != dim:
            raise ValueError(
                f"Existing index dim={cpu_index.d} != new dim={dim}. "
                "请确认使用的是同一 Embedding 模型 / 同一 prompt。"
            )
    add_index = cpu_index
    if use_gpu_if_possible and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        add_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    add_index.add(new_vecs.astype("float32"))
    try:
        cpu_index = faiss.index_gpu_to_cpu(add_index)
    except Exception:
        pass
    return cpu_index

def write_meta_jsonl(records: List[Dict[str, Any]], path: Path, base_id: int = 0, append: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for i, rec in enumerate(records):
            obj = dict(rec)
            obj.setdefault("id", base_id + i)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def file_blake2b_hex(data: bytes, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(data)
    return h.hexdigest()

def load_seen_doc_hashes(meta_path: Path) -> set[str]:
    seen: set[str] = set()
    p = Path(meta_path)
    if not p.exists():
        return seen
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                dh = obj.get("doc_hash")
                if dh:
                    seen.add(dh)
            except Exception:
                pass
    return seen

def _cli_ingest_docx(docx_file: str, model_name: str = "BAAI/bge-m3"):
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
    ap.add_argument("--model", default="BAAI/bge-m3")
    args = ap.parse_args()
    _cli_ingest_docx(args.docx, model_name=args.model)
