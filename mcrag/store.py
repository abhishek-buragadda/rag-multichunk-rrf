"""LanceDB storage: one table per chunk size."""

from __future__ import annotations

import json
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from .chunking import Chunk

META_FILE = "index_meta.json"


def table_name(size: int) -> str:
    return f"chunks_{size}"


class ChunkStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db = lancedb.connect(str(self.db_path))

    def write(self, size: int, chunks: list[Chunk], vectors: np.ndarray) -> None:
        dim = vectors.shape[1]
        schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("title", pa.string()),
                pa.field("text", pa.string()),
                pa.field("start", pa.int32()),
                pa.field("end", pa.int32()),
                pa.field("chunk_size", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        data = pa.table(
            {
                "chunk_id": [c.chunk_id for c in chunks],
                "doc_id": [c.doc_id for c in chunks],
                "title": [c.title for c in chunks],
                "text": [c.text for c in chunks],
                "start": [c.start for c in chunks],
                "end": [c.end for c in chunks],
                "chunk_size": [c.chunk_size for c in chunks],
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(vectors.astype(np.float32).ravel()), dim
                ),
            },
            schema=schema,
        )
        tbl = self.db.create_table(table_name(size), data=data, mode="overwrite")
        tbl.create_scalar_index("doc_id")

    def search(self, size: int, query_vec: np.ndarray, limit: int, where: str | None = None) -> list[dict]:
        tbl = self.db.open_table(table_name(size))
        q = tbl.search(query_vec.astype(np.float32)).distance_type("cosine")
        if where:
            q = q.where(where, prefilter=True)
        return (
            q.select(["chunk_id", "doc_id", "title", "text", "start", "end", "chunk_size", "_distance"])
            .limit(limit)
            .to_list()
        )

    def save_meta(self, meta: dict) -> None:
        (self.db_path / META_FILE).write_text(json.dumps(meta, indent=2))

    def load_meta(self) -> dict:
        path = self.db_path / META_FILE
        if not path.exists():
            raise FileNotFoundError(f"No index at {self.db_path}; run `python -m mcrag index` first.")
        return json.loads(path.read_text())
