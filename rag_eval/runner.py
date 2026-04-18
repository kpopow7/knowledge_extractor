from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag_eval.schema import EvalCase
from rag_index.search import SearchHit
from rag_index.targets import SearchIndexTarget
from rag_retrieve.pipeline import retrieve


def load_cases(path: Path) -> list[EvalCase]:
    rows: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(EvalCase.model_validate_json(line))
    return rows


@dataclass
class EvalMetrics:
    recall_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    n_labeled: int = 0


def _first_relevant_rank(hits: list[SearchHit], case: EvalCase) -> int | None:
    for r, h in enumerate(hits, start=1):
        pl = h.payload
        if case.relevance(
            h.chunk_id,
            pl.get("text_full") or "",
            int(pl.get("page_start") or 0),
            int(pl.get("page_end") or 0),
        ):
            return r
    return None


def run_evaluation(
    index_ref: SearchIndexTarget | Path,
    cases: list[EvalCase],
    *,
    ks: list[int] | None = None,
    final_k: int = 20,
    candidate_pool: int = 40,
    reranker: str = "none",
) -> tuple[list[dict], EvalMetrics]:
    if ks is None:
        ks = [1, 5, 10, 20]

    per_case: list[dict] = []
    mrr_sum = 0.0
    recall_hits = {k: 0 for k in ks}
    nl = 0

    for case in cases:
        hits = retrieve(
            index_ref,
            case.question,
            final_k=final_k,
            candidate_pool=candidate_pool,
            reranker=reranker,
        )
        labeled = bool(case.gold_chunk_ids or case.gold_pages or case.gold_substrings)
        rank = _first_relevant_rank(hits, case) if labeled else None
        per_case.append(
            {
                "id": case.id,
                "question": case.question,
                "first_relevant_rank": rank,
                "retrieved_chunk_ids": [h.chunk_id for h in hits[: max(ks)]],
            }
        )

        if not labeled:
            continue

        nl += 1
        if rank is not None:
            mrr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    recall_hits[k] += 1

    metrics = EvalMetrics(n_labeled=nl)
    metrics.mrr = mrr_sum / nl if nl else 0.0
    for k in ks:
        metrics.recall_at[k] = recall_hits[k] / nl if nl else 0.0

    return per_case, metrics


def metrics_to_json(metrics: EvalMetrics, ks: list[int]) -> dict:
    return {
        "n_labeled": metrics.n_labeled,
        "mrr": metrics.mrr,
        "recall": {f"@{k}": metrics.recall_at.get(k, 0.0) for k in ks},
    }
