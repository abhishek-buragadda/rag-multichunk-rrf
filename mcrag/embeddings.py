"""Local embedding model (no API key needed)."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge-*-v1.5 models retrieve better when queries (not passages) carry this instruction.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=_device())
        self.query_prefix = QUERY_PREFIXES.get(model_name, "")

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            [self.query_prefix + t for t in texts], batch_size=64, normalize_embeddings=True
        )
