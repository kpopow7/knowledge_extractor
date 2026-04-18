from __future__ import annotations

import argparse

import rag_storage  # noqa: F401 — loads .env before RAG_API_HOST / PORT defaults
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the RAG HTTP API (FastAPI + uvicorn).")
    p.add_argument("--host", default=os.environ.get("RAG_API_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("RAG_API_PORT", "8000")))
    p.add_argument(
        "--reload",
        action="store_true",
        help="Dev-only: reload on code changes (do not use in production).",
    )
    args = p.parse_args(argv)

    try:
        import uvicorn
    except ImportError as e:
        print("Install uvicorn: pip install uvicorn[standard]", file=sys.stderr)
        raise SystemExit(1) from e

    uvicorn.run(
        "rag_api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
