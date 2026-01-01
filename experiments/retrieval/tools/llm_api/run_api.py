import argparse, json, os, re, time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI

JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
CLAUSE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{0,32}_[0-9]+(?:\.[0-9]+)*")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_done_qids(out_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = str(obj.get("qid", ""))
                if qid:
                    done.add(qid)
            except Exception:
                continue
    return done

def extract_clause_ids(text: str, k: int) -> List[str]:
    text = (text or "").strip()
    # 1) JSON parse
    m = JSON_RE.search(text)
    if m:
        blob = m.group(1)
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and "clause_ids" in obj and isinstance(obj["clause_ids"], list):
                ids = [str(x).strip() for x in obj["clause_ids"] if str(x).strip()]
                return dedup(ids)[:k]
            if isinstance(obj, list):
                ids = [str(x).strip() for x in obj if str(x).strip()]
                return dedup(ids)[:k]
        except Exception:
            pass
    # 2) regex fallback
    ids = CLAUSE_RE.findall(text)
    return dedup(ids)[:k]

def dedup(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def build_prompt(template: str, query: str, k: int) -> str:
    return template.format(query=query, k=k)

@retry(reraise=True, wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def call_openai_responses(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
                         temperature: float, max_tokens: int) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    usage = getattr(resp, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "input_tokens", None),
            "completion_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return {"text": getattr(resp, "output_text", ""), "usage": usage_dict}

@retry(reraise=True, wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def call_openai_chatcompletions(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
                                temperature: float, max_tokens: int) -> Dict[str, Any]:
    comp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = comp.choices[0].message.content if comp.choices else ""
    usage = getattr(comp, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return {"text": text, "usage": usage_dict}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "dashscope"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="queries jsonl")
    ap.add_argument("--output", required=True, help="output jsonl")
    ap.add_argument("--prompt", required=True, help="prompt template file")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.prompt, "r", encoding="utf-8") as f:
        template = f.read()

    system_prompt = "You are a precise assistant that follows output format strictly."
    done = load_done_qids(args.output)

    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        caller = call_openai_responses
    else:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        caller = call_openai_chatcompletions

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n = 0
    with open(args.output, "a", encoding="utf-8") as out_f:
        for row in iter_jsonl(args.input):
            qid = str(row.get("qid") or row.get("id") or row.get("query_id") or "")
            query = row.get("query") or row.get("question") or row.get("q") or row.get("text") or ""
            if not qid:
                # fallback: stable index-based id
                qid = f"idx_{n}"
            if qid in done:
                continue

            user_prompt = build_prompt(template, str(query), args.k)

            t0 = time.perf_counter()
            err = None
            text = ""
            usage = None
            try:
                res = caller(client, args.model, system_prompt, user_prompt, args.temperature, args.max_tokens)
                text = res.get("text") or ""
                usage = res.get("usage")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            latency_s = time.perf_counter() - t0
            clause_ids = extract_clause_ids(text, args.k) if not err else []

            rec = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "provider": args.provider,
                "model": args.model,
                "qid": qid,
                "query": query,
                "pred_clause_ids": clause_ids,
                "raw_text": text,
                "latency_s": latency_s,
                "usage": usage,
                "error": err,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            done.add(qid)
            n += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            if args.limit and n >= args.limit:
                break

if __name__ == "__main__":
    main()
