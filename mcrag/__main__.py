"""CLI: python -m mcrag {index,query,eval} ..."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from .corpus import load_beir, load_folder
from .embeddings import DEFAULT_MODEL
from .pipeline import DEFAULT_SIZES, Retriever, build_index

DATA_DIR = Path("data")


def _db_path(args) -> Path:
    return DATA_DIR / "lancedb" / (args.dataset if not getattr(args, "docs", None) else "local")


def cmd_index(args):
    if args.docs:
        corpus, source = load_folder(Path(args.docs)), f"folder:{args.docs}"
    else:
        corpus, source = load_beir(args.dataset, DATA_DIR / "beir"), f"beir:{args.dataset}"
    build_index(corpus, _db_path(args), args.sizes, args.model, args.overlap, source)
    print(f"Index written to {_db_path(args)}")


def cmd_query(args):
    retriever = Retriever(_db_path(args))
    if args.chunks:
        for i, (chunk, score) in enumerate(retriever.retrieve_chunks(args.text, args.top_k), start=1):
            print(f"\n#{i}  score={score:.4f}  doc={chunk['doc_id']}  size={chunk['chunk_size']}")
            print(textwrap.indent(textwrap.fill(chunk["text"], 100), "    "))
        return
    for i, hit in enumerate(retriever.retrieve(args.text, top_k=args.top_k), start=1):
        ranks = ", ".join(f"{s}w:#{r}" for s, r in sorted(hit.ranks.items()))
        print(f"\n#{i}  rrf={hit.score:.4f}  doc={hit.doc_id}  [{ranks}]")
        print(f"    {hit.title}")
        snippet = hit.best_chunk["text"][:400]
        print(textwrap.indent(textwrap.fill(snippet, 100), "    > "))


def cmd_eval(args):
    from .evaluate import evaluate, print_report

    corpus = load_beir(args.dataset, DATA_DIR / "beir")
    retriever = Retriever(_db_path(args))
    print_report(evaluate(retriever, corpus, per_size_k=args.per_size_k, rrf_k=args.rrf_k))


def cmd_bench(args):
    from .evaluate import evaluate, print_report
    from .long_corpora import load_qasper, load_seinfeld

    long_dir = DATA_DIR / "long"
    db = DATA_DIR / "lancedb" / args.name
    if args.name == "qasper":
        corpus = load_qasper(long_dir)
    else:
        corpus = load_seinfeld(long_dir, Path(args.scripts_csv) if args.scripts_csv else None)
    print(f"{args.name}: {len(corpus.docs)} docs, {len(corpus.queries)} queries")

    if args.reindex or not (db / "index_meta.json").exists():
        # No title prefix: for Seinfeld the episode title would leak the answer into every chunk.
        build_index(corpus, db, args.sizes, args.model, source=f"bench:{args.name}", prepend_title=False)
    retriever = Retriever(db)

    if args.name == "qasper":
        from .passage_eval import evaluate_passages

        print_report(evaluate_passages(retriever, corpus), fused=("RRF(interleave)", "RRF(span)"))
    else:
        print_report(evaluate(retriever, corpus, per_size_k=args.per_size_k))


def main():
    p = argparse.ArgumentParser(prog="mcrag", description=__doc__)
    sub = p.add_subparsers(required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")

    pi = sub.add_parser("index", parents=[common], help="chunk, embed and store the corpus")
    pi.add_argument("--docs", help="index a local folder of .txt/.md files instead of a BEIR dataset")
    pi.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES), help="chunk sizes in words")
    pi.add_argument("--overlap", type=float, default=0.15)
    pi.add_argument("--model", default=DEFAULT_MODEL)
    pi.set_defaults(func=cmd_index)

    pq = sub.add_parser("query", parents=[common], help="retrieve with multi-size RRF")
    pq.add_argument("text")
    pq.add_argument("--docs", action="store_true", help="query the local-folder index")
    pq.add_argument("--top-k", type=int, default=5)
    pq.add_argument("--chunks", action="store_true", help="return fused chunks instead of documents")
    pq.set_defaults(func=cmd_query)

    pe = sub.add_parser("eval", parents=[common], help="single sizes vs RRF on BEIR qrels")
    pe.add_argument("--per-size-k", type=int, default=100)
    pe.add_argument("--rrf-k", type=int, default=60)
    pe.set_defaults(func=cmd_eval)

    pb = sub.add_parser("bench", help="long-document benchmarks: qasper (passages) or seinfeld (episodes)")
    pb.add_argument("name", choices=["qasper", "seinfeld"])
    pb.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    pb.add_argument("--model", default=DEFAULT_MODEL)
    pb.add_argument("--reindex", action="store_true")
    pb.add_argument("--per-size-k", type=int, default=200)
    pb.add_argument("--scripts-csv", help="Kaggle Seinfeld scripts.csv (default: Hugging Face mirror)")
    pb.set_defaults(func=cmd_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
