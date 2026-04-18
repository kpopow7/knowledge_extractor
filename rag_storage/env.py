"""Load a project-root ``.env`` into the process environment (optional dependency)."""

from __future__ import annotations

from pathlib import Path


def load_dotenv_from_project() -> None:
    """
    Load ``.env`` from the repository root, then the current working directory.

    Does not override variables already set in the environment. If ``python-dotenv``
    is not installed, this is a no-op.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    load_dotenv()
