"""Corpus loaders.

Two sources are supported:

* BEIR benchmark datasets (default: SciFact). These ship with queries and
  relevance judgements (qrels), so we can measure whether multi-chunk RRF
  actually beats a single chunk size.
* A local folder of .txt / .md files, for your own documents (no evaluation).
"""

from __future__ import annotations

import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Corpus:
    docs: list[Document]
    queries: dict[str, str] = field(default_factory=dict)
    # query_id -> {doc_id: relevance}
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)


def download_beir(name: str, data_dir: Path) -> Path:
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    url = BEIR_URL.format(name=name)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
    return target


def load_beir(name: str, data_dir: Path, split: str = "test") -> Corpus:
    root = download_beir(name, data_dir)

    docs = []
    with open(root / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(str(row["_id"]), row.get("title", ""), row["text"]))

    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)

    queries = {}
    with open(root / "queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if str(row["_id"]) in qrels:
                queries[str(row["_id"])] = row["text"]

    return Corpus(docs, queries, qrels)


def load_folder(path: Path) -> Corpus:
    docs = []
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in {".txt", ".md"} and p.is_file():
            docs.append(Document(str(p.relative_to(path)), p.stem, p.read_text(errors="ignore")))
    if not docs:
        raise ValueError(f"No .txt/.md files found under {path}")
    return Corpus(docs)
