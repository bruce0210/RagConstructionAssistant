# core/utils/oss_io.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple
import oss2

def _get(s: Optional[dict], key: str, env: str, default: str = "") -> str:
    if s and key in s and s[key]:
        return str(s[key])
    return os.getenv(env, default)

def get_oss_clients(secrets: Optional[dict] = None):
    """
    返回:
      - bucket_docx_puburl, bucket_media_puburl, bucket_index_puburl: 生成公网URL的辅助闭包
      - bucket_docx, bucket_media, bucket_index: 内网上传用的 Bucket 客户端
    """
    sec = secrets.get("oss") if secrets else None

    ep_internal = _get(sec, "endpoint_internal", "OSS_ENDPOINT_INTERNAL")
    ep_public   = _get(sec, "endpoint_public",   "OSS_ENDPOINT_PUBLIC")

    ak = _get(sec, "access_key_id",     "OSS_ACCESS_KEY_ID")
    sk = _get(sec, "access_key_secret", "OSS_ACCESS_KEY_SECRET")

    b_docx  = _get(sec, "bucket_docx",  "OSS_BUCKET_DOCX")
    b_media = _get(sec, "bucket_media", "OSS_BUCKET_MEDIA")
    b_index = _get(sec, "bucket_index", "OSS_BUCKET_INDEX")

    pub_docx  = _get(sec, "public_base_docx",  "OSS_PUBLIC_BASE_DOCX")
    pub_media = _get(sec, "public_base_media", "OSS_PUBLIC_BASE_MEDIA")
    pub_index = _get(sec, "public_base_index", "OSS_PUBLIC_BASE_INDEX")

    if not (ep_internal and ep_public and ak and sk and b_docx and b_media):
        raise RuntimeError("OSS 配置不完整（检查 endpoint_internal/endpoint_public/ak/sk/bucket_docx/bucket_media）")

    auth = oss2.Auth(ak, sk)
    bucket_docx  = oss2.Bucket(auth, ep_internal, b_docx)
    bucket_media = oss2.Bucket(auth, ep_internal, b_media)
    bucket_index = oss2.Bucket(auth, ep_internal, b_index) if b_index else None

    def _make_url(public_base: str, endpoint_public: str, bucket_name: str, key: str) -> str:
        if public_base:
            return f"{public_base.rstrip('/')}/{key}"
        host = endpoint_public.replace("https://", "").replace("http://", "")
        return f"https://{bucket_name}.{host}/{key}"

    def _url_docx(key: str) -> str:
        return _make_url(pub_docx, ep_public, b_docx, key)

    def _url_media(key: str) -> str:
        return _make_url(pub_media, ep_public, b_media, key)

    def _url_index(key: str) -> str:
        if not b_index:
            return ""
        return _make_url(pub_index, ep_public, b_index, key)

    return (bucket_docx, bucket_media, bucket_index, _url_docx, _url_media, _url_index)

def oss_put(bucket: oss2.Bucket, local_path: Path, key: str):
    with open(local_path, "rb") as f:
        bucket.put_object(key, f)
