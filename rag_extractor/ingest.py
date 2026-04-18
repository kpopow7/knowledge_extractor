from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rag_extractor import EXTRACTION_VERSION
from rag_extractor.extract import _sha256_file, extract_pdf, write_artifact
from rag_extractor.paths import documents_dir, storage_root
from rag_extractor.registry import DocumentRegistry
from rag_storage.blob import write_blob
from rag_storage.config import use_s3_blobs


SOURCE_NAME = "source.pdf"
ARTIFACT_NAME = "extraction.json"


def _doc_dir(content_sha256: str) -> Path:
    return documents_dir() / content_sha256


def _mirror_to_object_storage(source_relpath: str, artifact_relpath: str | None) -> None:
    """If ``RAG_S3_BUCKET`` is set, copy files under ``RAG_STORAGE_ROOT`` to the bucket."""
    if not use_s3_blobs():
        return
    root = storage_root()
    write_blob(source_relpath, (root / source_relpath).read_bytes())
    if artifact_relpath and (root / artifact_relpath).is_file():
        write_blob(artifact_relpath, (root / artifact_relpath).read_bytes())


def _rel_to_storage(path: Path) -> str:
    root = storage_root().resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@dataclass
class IngestResult:
    content_sha256: str
    skipped: bool
    reason: str | None
    source_path: Path
    artifact_path: Path | None
    status: str


def ingest_pdf(
    pdf_path: str | Path,
    *,
    force: bool = False,
    registry: DocumentRegistry | None = None,
) -> IngestResult:
    """
    Copy PDF into storage/documents/<sha256>/, run extraction, register in SQLite.
    Idempotent: same file hash + same EXTRACTION_VERSION skips work unless force=True.
    """
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))

    reg = registry or DocumentRegistry()

    content_sha256 = _sha256_file(path)
    byte_size = path.stat().st_size

    ddir = _doc_dir(content_sha256)
    source_abs = ddir / SOURCE_NAME
    artifact_abs = ddir / ARTIFACT_NAME
    source_rel = _rel_to_storage(source_abs)
    artifact_rel = _rel_to_storage(artifact_abs)

    existing = reg.get(content_sha256)
    if (
        existing
        and existing.status == "ready"
        and existing.extraction_version == EXTRACTION_VERSION
        and not force
    ):
        ap = storage_root() / existing.artifact_relpath if existing.artifact_relpath else artifact_abs
        sp = storage_root() / existing.source_relpath
        return IngestResult(
            content_sha256=content_sha256,
            skipped=True,
            reason="already_ingested_same_version",
            source_path=sp,
            artifact_path=ap if ap.is_file() else None,
            status="ready",
        )

    ddir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, source_abs)
    reg.upsert_pending(
        content_sha256=content_sha256,
        original_filename=path.name,
        byte_size=byte_size,
        source_relpath=source_rel,
    )

    try:
        artifact = extract_pdf(source_abs, document_id=content_sha256)
        write_artifact(artifact, artifact_abs)
        reg.mark_ready(
            content_sha256=content_sha256,
            extraction_version=artifact.extraction_version,
            extractor_package_version=artifact.extractor_package_version,
            schema_version=artifact.schema_version,
            page_count=artifact.page_count,
            artifact_relpath=artifact_rel,
        )
        _mirror_to_object_storage(source_rel, artifact_rel)
    except Exception as e:  # noqa: BLE001
        reg.mark_failed(content_sha256=content_sha256, error_message=f"{type(e).__name__}: {e}")
        return IngestResult(
            content_sha256=content_sha256,
            skipped=False,
            reason=None,
            source_path=source_abs,
            artifact_path=None,
            status="failed",
        )

    return IngestResult(
        content_sha256=content_sha256,
        skipped=False,
        reason=None,
        source_path=source_abs,
        artifact_path=artifact_abs,
        status="ready",
    )
