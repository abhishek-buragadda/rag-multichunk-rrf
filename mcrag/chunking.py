"""Sentence-aware chunking at a configurable size (measured in words)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start: int  # char offset in the original document text
    end: int
    chunk_size: int


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _split_long(text: str, start: int, end: int, max_words: int) -> list[tuple[int, int]]:
    """Split a single over-long sentence on word boundaries."""
    words = [(m.start() + start, m.end() + start) for m in re.finditer(r"\S+", text[start:end])]
    return [
        (words[i][0], words[min(i + max_words, len(words)) - 1][1])
        for i in range(0, len(words), max_words)
    ]


def chunk_document(doc: Document, size: int, overlap: float = 0.15) -> list[Chunk]:
    """Greedily pack sentences into chunks of at most `size` words.

    Consecutive chunks share roughly `overlap * size` words of trailing
    sentences so that facts on a boundary are not cut in half.
    """
    units: list[tuple[int, int, int]] = []  # (start, end, n_words)
    for s, e in _sentence_spans(doc.text):
        n = len(doc.text[s:e].split())
        if n > size:
            for ss, ee in _split_long(doc.text, s, e, size):
                units.append((ss, ee, len(doc.text[ss:ee].split())))
        else:
            units.append((s, e, n))

    overlap_words = int(size * overlap)
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j, words = i, 0
        while j < len(units) and words + units[j][2] <= size:
            words += units[j][2]
            j += 1
        j = max(j, i + 1)
        start, end = units[i][0], units[j - 1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{size}::{len(chunks)}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=doc.text[start:end],
                start=start,
                end=end,
                chunk_size=size,
            )
        )
        if j >= len(units):
            break
        # Step back over trailing sentences to create the overlap, but always advance.
        back, k = 0, j
        while k - 1 > i and back + units[k - 1][2] <= overlap_words:
            back += units[k - 1][2]
            k -= 1
        i = k
    return chunks
