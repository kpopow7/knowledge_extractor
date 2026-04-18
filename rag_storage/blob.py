from __future__ import annotations

import os
from pathlib import Path

from rag_extractor.paths import storage_root
from rag_storage.config import s3_bucket, s3_key_prefix, use_s3_blobs


def _local_path(rel: str) -> Path:
    rel = rel.replace("\\", "/").lstrip("/")
    return storage_root() / rel


def write_blob(rel_path: str, data: bytes) -> None:
    """Write bytes to ``<storage_root>/<rel_path>`` or S3 when configured."""
    if use_s3_blobs():
        _s3_put(rel_path, data)
        return
    p = _local_path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def read_blob(rel_path: str) -> bytes:
    if use_s3_blobs():
        return _s3_get(rel_path)
    p = _local_path(rel_path)
    return p.read_bytes()


def blob_exists(rel_path: str) -> bool:
    if use_s3_blobs():
        return _s3_head(rel_path)
    return _local_path(rel_path).is_file()


def ensure_dir_for_local(rel_path: str) -> None:
    """Create parent dirs for local storage (no-op for pure S3)."""
    if not use_s3_blobs():
        _local_path(rel_path).parent.mkdir(parents=True, exist_ok=True)


def _object_key(rel_path: str) -> str:
    rel = rel.replace("\\", "/").lstrip("/")
    pref = s3_key_prefix()
    return f"{pref}/{rel}" if pref else rel


def _s3_client():
    import boto3

    kwargs = {}
    endpoint = (os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL") or "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
        **kwargs,
    )


def _s3_put(rel_path: str, data: bytes) -> None:
    client = _s3_client()
    client.put_object(Bucket=s3_bucket(), Key=_object_key(rel_path), Body=data)


def _s3_get(rel_path: str) -> bytes:
    client = _s3_client()
    r = client.get_object(Bucket=s3_bucket(), Key=_object_key(rel_path))
    return r["Body"].read()


def _s3_head(rel_path: str) -> bool:
    import botocore.exceptions

    client = _s3_client()
    try:
        client.head_object(Bucket=s3_bucket(), Key=_object_key(rel_path))
        return True
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
