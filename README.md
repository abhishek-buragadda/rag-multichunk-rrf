# rag-multichunk-rrf
Using variable size chunks for creating embeddings for better RAG performance.

Each document is chunked at several sizes (default 64 / 128 / 256 words, sentence-aware, 15% overlap),
embedded locally with `BAAI/bge-small-en-v1.5`, and stored in **LanceDB** (one table per size).
At query time every table is searched, each chunk ranking is collapsed to a document ranking
(a doc's rank = rank of its best chunk), and the per-size rankings are merged with
**Reciprocal Rank Fusion**: `score(d) = Σ 1 / (k + rank_size(d))`, `k = 60`.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Usage
```bash
# downloads BEIR SciFact (5,183 abstracts, 300 test queries with relevance labels) and indexes it
.venv/bin/python -m mcrag index
.venv/bin/python -m mcrag query "Vitamin D deficiency increases risk of multiple sclerosis"
.venv/bin/python -m mcrag query "..." --chunks        # fused passages for an LLM prompt
.venv/bin/python -m mcrag eval                        # single sizes vs RRF

# other BEIR sets: --dataset nfcorpus | fiqa | trec-covid | ...
# your own files:  index --docs path/to/folder   then   query "..." --docs
```

## Results (SciFact test, bge-small-en-v1.5)
| system         | nDCG@10 | MRR@10 | R@10   | R@100  |
|----------------|---------|--------|--------|--------|
| size=64        | 0.7228  | 0.6906 | 0.8363 | 0.9210 |
| size=128       | 0.7194  | 0.6909 | 0.8279 | 0.9433 |
| size=256       | 0.7150  | 0.6861 | 0.8329 | 0.9417 |
| RRF(all sizes) | 0.7222  | 0.6892 | **0.8412** | **0.9467** |

RRF gives slightly better recall and matches the best single size on ranking quality. The differences
are within noise: SciFact abstracts (~200 words) are too short for chunk size to matter.

## Long documents
```bash
.venv/bin/python -m mcrag bench seinfeld   # 176 scripts (~4k words), 757 queries
.venv/bin/python -m mcrag bench qasper     # 416 NLP papers (~3.4k words), 1,289 questions
```
p-values come from a paired permutation test against the *best* single size for each metric
(picked after the fact, which favours the single size).

### Seinfeld: which episode is this? (document-level)
Scripts: [Kaggle Seinfeld Chronicles](https://www.kaggle.com/datasets/thec03u5/seinfeld-chronicles), loaded from the
`Adam173/seinfeld-scripts` Hugging Face mirror (pass `--scripts-csv` to use the Kaggle CSV).
Queries: each sentence of the episode's Wikipedia plot summary. Target: that episode.

| system         | nDCG@10 | MRR@10 | Hit@1  | R@10   |
|----------------|---------|--------|--------|--------|
| size=64        | 0.6927  | 0.6524 | 0.5509 | 0.8369 |
| size=128       | 0.6738  | 0.6346 | 0.5390 | 0.8190 |
| size=256       | 0.6103  | 0.5738 | 0.4901 | 0.7503 |
| RRF(all sizes) | **0.7322** | **0.6979** | **0.6103** | **0.8600** |

RRF vs best single size: Hit@1 +5.9 pts, MRR +4.5 pts (p < 0.001).

### QASPER: evidence recall in a fixed context budget (passage-level)
Retrieval runs inside the question's paper. Chunks fill an N-word context in rank order; the score is the share
of annotated evidence paragraphs that make it in (see `mcrag/passage_eval.py`).

| system          | 128w   | 256w   | 512w   | 1024w  |
|-----------------|--------|--------|--------|--------|
| size=64         | 0.2397 | 0.3957 | 0.5705 | 0.7558 |
| size=128        | 0.2968 | 0.4281 | 0.5956 | 0.7635 |
| size=256        | 0.2088 | 0.3309 | 0.5116 | 0.7124 |
| RRF(interleave) | 0.2971 | 0.4285 | 0.6030 | 0.7807 |
| RRF(span)       | 0.2805 | **0.4523** | **0.6083** | **0.7834** |

Plain RRF over chunk ids just interleaves the sizes and ties the best single size. Overlap-aware RRF
(each word scored by the ranks of the chunks containing it) gains about 2 pts at 256w and 1024w (p ≈ 0.02),
but loses 1.6 pts at 128w (not significant).

### Conclusion
- Short documents: no benefit. Use one chunk size.
- Long documents, finding the right document: clear, significant win.
- Long documents, choosing passages for an LLM: small gain, and only with overlap-aware fusion.
- Cost: 1.75× the vectors of the 64-word index alone (3.5× vs 128 alone) and 3 searches per query.

## Layout
- `mcrag/corpus.py`: BEIR + local-folder loaders
- `mcrag/chunking.py`: sentence-aware chunker with character offsets
- `mcrag/embeddings.py`: sentence-transformers wrapper (MPS/CUDA/CPU)
- `mcrag/store.py`: LanceDB tables per chunk size
- `mcrag/fusion.py`: RRF
- `mcrag/pipeline.py`: indexing + retrieval
- `mcrag/evaluate.py`: nDCG / MRR / Recall + paired permutation test
- `mcrag/long_corpora.py`: QASPER and Seinfeld (+ Wikipedia summaries as queries) loaders
- `mcrag/passage_eval.py`: passage-level evidence recall and overlap-aware RRF
