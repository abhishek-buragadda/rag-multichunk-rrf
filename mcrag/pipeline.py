"""Index documents at several chunk sizes and retrieve with RRF."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking import chunk_document
from .corpus import Corpus
from .embeddings import DEFAULT_MODEL, Embedder
from .fusion import dedupe_preserving_order, reciprocal_rank_fusion
from .store import ChunkStore

DEFAULT_SIZES = (64, 128, 256)


# this is where different size chunks are created and stored into lance.
def build_index(
    corpus: Corpus,
    db_path: Path,
    sizes=DEFAULT_SIZES,
    model_name: str = DEFAULT_MODEL,
    overlap: float = 0.15,
    source: str = "",
    prepend_title: bool = True,
) -> None:
    embedder = Embedder(model_name)
    store = ChunkStore(db_path)
    counts = {}
    for size in sizes:
        chunks = [c for d in corpus.docs for c in chunk_document(d, size, overlap)]
        print(f"[size={size}] {len(chunks)} chunks from {len(corpus.docs)} docs; embedding...")
        # Prepend the title so short chunks keep their document context.
        texts = [f"{c.title}\n{c.text}" if c.title and prepend_title else c.text for c in chunks]
        store.write(size, chunks, embedder.embed_passages(texts))
        counts[size] = len(chunks)
    store.save_meta(
        {"model": model_name, "sizes": list(sizes), "overlap": overlap, "source": source, "chunks": counts}
    )


@dataclass
class Hit:
    doc_id: str
    score: float
    title: str
    best_chunk: dict
    # chunk size -> rank of this doc in that size's list (1-based), if it appeared
    ranks: dict[int, int] = field(default_factory=dict)


class Retriever:
    def __init__(self, db_path: Path, embedder: Embedder | None = None):
        self.store = ChunkStore(db_path)
        self.meta = self.store.load_meta()
        self.sizes: list[int] = self.meta["sizes"]
        self.embedder = embedder or Embedder(self.meta["model"])

    def search_all_sizes(
        self, query_vec: np.ndarray, per_size_k: int, where: str | None = None
    ) -> dict[int, list[dict]]:
        return {s: self.store.search(s, query_vec, per_size_k, where) for s in self.sizes}

    def retrieve(
        self,
        query: str | None = None,
        *,
        query_vec: np.ndarray | None = None,
        top_k: int = 10,
        per_size_k: int = 50,
        rrf_k: int = 60,
        sizes: list[int] | None = None,
    ) -> list[Hit]:
        """Document-level RRF across chunk sizes.

        Chunks of different sizes never share an id, so fusing raw chunk ids
        would just interleave the lists. Instead each size's chunk ranking is
        collapsed to a document ranking (a doc's rank = rank of its best chunk)
        and those document rankings are fused.
        """
        if query_vec is None:
            query_vec = self.embedder.embed_queries([query])[0]
        per_size = self.search_all_sizes(query_vec, per_size_k)
        if sizes is not None:
            per_size = {s: per_size[s] for s in sizes}

        doc_lists = {s: dedupe_preserving_order([r["doc_id"] for r in rows]) for s, rows in per_size.items()}
        fused = reciprocal_rank_fusion({str(s): docs for s, docs in doc_lists.items()}, k=rrf_k)

        # Best supporting chunk per doc = its highest-ranked chunk in any list.
        best: dict[str, tuple[int, dict]] = {}
        for rows in per_size.values():
            for rank, r in enumerate(rows, start=1):
                if r["doc_id"] not in best or rank < best[r["doc_id"]][0]:
                    best[r["doc_id"]] = (rank, r)

        hits = []
        for doc_id, score in fused[:top_k]:
            chunk = best[doc_id][1]
            ranks = {s: docs.index(doc_id) + 1 for s, docs in doc_lists.items() if doc_id in docs}
            hits.append(Hit(doc_id, score, chunk["title"], chunk, ranks))
        return hits

    def retrieve_chunks(
        self, query: str, top_k: int = 10, per_size_k: int = 50, rrf_k: int = 60
    ) -> list[tuple[dict, float]]:
        """Chunk-level context for an LLM prompt.

        A chunk's score is the RRF score of the document it belongs to plus
        its own reciprocal rank within its list, so passages from strongly
        fused documents come first while each size can still contribute.
        """
        query_vec = self.embedder.embed_queries([query])[0]
        per_size = self.search_all_sizes(query_vec, per_size_k)
        doc_lists = {str(s): dedupe_preserving_order([r["doc_id"] for r in rows]) for s, rows in per_size.items()}
        doc_score = dict(reciprocal_rank_fusion(doc_lists, k=rrf_k))
        scored = [
            (r, doc_score[r["doc_id"]] + 1.0 / (rrf_k + rank))
            for rows in per_size.values()
            for rank, r in enumerate(rows, start=1)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
