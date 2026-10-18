from mcrag.chunking import chunk_document
from mcrag.corpus import Document
from mcrag.evaluate import ndcg_at_k, recall_at_k
from mcrag.fusion import reciprocal_rank_fusion


def test_rrf_rewards_consensus():
    fused = reciprocal_rank_fusion({"a": ["x", "y", "z"], "b": ["y", "x", "w"], "c": ["y", "q"]}, k=60)
    assert [d for d, _ in fused][:2] == ["y", "x"]
    assert abs(dict(fused)["y"] - (1 / 62 + 1 / 61 + 1 / 61)) < 1e-12


def test_rrf_weights():
    fused = reciprocal_rank_fusion({"a": ["x"], "b": ["y"]}, weights={"b": 2.0})
    assert fused[0][0] == "y"


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"Sentence number {i} has some words in it." for i in range(100))
    doc = Document("d1", "T", text)
    for size in (16, 64, 256):
        chunks = chunk_document(doc, size)
        assert all(len(c.text.split()) <= size for c in chunks)
        assert all(doc.text[c.start : c.end] == c.text for c in chunks)
        assert chunks[0].start == 0 and chunks[-1].end == len(text)
        # contiguous or overlapping, never a gap
        assert all(b.start <= a.end + 1 for a, b in zip(chunks, chunks[1:]))


def test_long_sentence_is_split():
    doc = Document("d", "", " ".join(["word"] * 300))
    chunks = chunk_document(doc, 64)
    assert len(chunks) >= 5 and all(len(c.text.split()) <= 64 for c in chunks)


def test_metrics():
    rels = {"a": 1, "b": 1}
    assert recall_at_k(["a", "x", "b"], rels, 2) == 0.5
    assert ndcg_at_k(["a", "b"], rels, 10) == 1.0
