from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def run_with_timeout(
    fn: Callable[[], T],
    timeout_sec: float | None,
    *,
    label: str = "task",
) -> T:
    """
    Run a blocking callable in a worker thread with an overall timeout (cross-platform).
    If ``timeout_sec`` is None or <= 0, runs ``fn()`` directly (no thread hop).
    """
    if timeout_sec is None or timeout_sec <= 0:
        return fn()

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        done, _not_done = wait([fut], timeout=timeout_sec)
        if not done:
            log.warning("%s exceeded timeout %.2fs", label, timeout_sec)
            raise TimeoutError(f"{label} exceeded timeout ({timeout_sec}s)")
        return fut.result()
