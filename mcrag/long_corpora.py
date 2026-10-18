"""Long-document benchmarks: QASPER (NLP papers) and Seinfeld scripts.

QASPER  - questions about a single paper, with annotated evidence paragraphs.
          Used for passage-level evaluation inside the paper.
Seinfeld - 176 episode scripts. There are no labels in the Kaggle dump, so queries
          are sentences from each episode's Wikipedia plot summary and the target
          is that episode (document-level, corpus-wide retrieval).
"""

from __future__ import annotations

import re
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from .corpus import Corpus, Document

QASPER_URL = "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/{split}/0.parquet"
# Mirror of the Kaggle "Seinfeld Chronicles" scripts (Kaggle needs an account to download).
SEINFELD_URL = "https://huggingface.co/api/datasets/Adam173/seinfeld-scripts/parquet/default/train/0.parquet"
WIKI_RAW = "https://en.wikipedia.org/w/index.php?title={page}&action=raw"
USER_AGENT = "mcrag-benchmark/0.1 (research script)"


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as r:
            dest.write_bytes(r.read())
    return dest


# ---------------------------------------------------------------- QASPER


@dataclass
class QasperCorpus(Corpus):
    # query_id -> doc_id of the paper it was asked about
    query_doc: dict[str, str] = field(default_factory=dict)
    # query_id -> one list of evidence char spans per annotator
    evidence: dict[str, list[list[tuple[int, int]]]] = field(default_factory=dict)


def load_qasper(data_dir: Path, split: str = "test") -> QasperCorpus:
    path = _fetch(QASPER_URL.format(split=split), data_dir / f"qasper_{split}.parquet")
    corpus = QasperCorpus(docs=[])
    for paper in pq.read_table(path).to_pylist():
        parts, para_span, pos = [], {}, 0

        def add(text: str, is_para: bool):
            nonlocal pos
            if is_para:
                para_span.setdefault(text.strip(), (pos, pos + len(text)))
            parts.append(text)
            pos += len(text) + 2  # "\n\n" separator

        add(paper["abstract"], True)
        ft = paper["full_text"]
        for name, paras in zip(ft["section_name"], ft["paragraphs"]):
            if name:
                add(name, False)
            for p in paras:
                if p.strip():
                    add(p, True)
        doc_id = paper["id"]
        corpus.docs.append(Document(doc_id, paper["title"], "\n\n".join(parts)))

        qas = paper["qas"]
        for q, qid, ans in zip(qas["question"], qas["question_id"], qas["answers"]):
            annotator_spans = []
            for a in ans["answer"]:
                if a["unanswerable"] or not a["evidence"]:
                    continue
                # Evidence from tables/figures ("FLOAT SELECTED") is not in the text; skip it.
                spans = [para_span.get(e.strip()) for e in a["evidence"] if "FLOAT SELECTED" not in e]
                if spans and all(spans):
                    annotator_spans.append(spans)
            if annotator_spans:
                corpus.queries[qid] = q.strip()
                corpus.query_doc[qid] = doc_id
                corpus.evidence[qid] = annotator_spans
    return corpus


# ---------------------------------------------------------------- Seinfeld


# Script titles that differ from their Wikipedia names.
_TITLE_ALIASES = {"GOODNEWSBADNEWS": "SEINFELDCHRONICLES"}


def _norm_title(t: str) -> str:
    """'The Stake Out', 'THE STAKEOUT', 'The Trip (Part 1)' -> 'STAKEOUT', 'TRIP'."""
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().upper()
    t = re.sub(r"\(.*?\)|\bPART \d\b|\bAND\b|^THE\b", "", t.strip())
    t = re.sub(r"[^A-Z0-9]", "", t)
    return _TITLE_ALIASES.get(t, t)


def _clean_wikitext(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"\s*\([^()]*\)", "", s)  # actor names in parentheses
    return " ".join(s.split())


def _wiki_summaries(data_dir: Path) -> dict[str, str]:
    """Normalized episode title -> Wikipedia ShortSummary, for seasons 1-9."""
    out = {}
    for season in range(1, 10):
        page = f"Seinfeld_season_{season}"
        raw = _fetch(WIKI_RAW.format(page=page), data_dir / "wiki" / f"{page}.txt").read_text()
        for block in raw.split("{{Episode list")[1:]:
            title = re.search(r"\|\s*Title\s*=\s*(.+)", block)
            summ = re.search(r"\|\s*ShortSummary\s*=\s*(.+?)(?=\n\s*\|\s*\w+\s*=|\n\}\})", block, re.S)
            if title and summ:
                out[_norm_title(_clean_wikitext(title.group(1)))] = _clean_wikitext(summ.group(1))
    return out


def load_seinfeld(data_dir: Path, scripts_csv: Path | None = None) -> Corpus:
    """Scripts from the HF mirror, or from Kaggle's scripts.csv + episode_info.csv if given."""
    docs = []
    if scripts_csv:
        import csv

        info = {r["SEID"]: r["Title"] for r in csv.DictReader(open(scripts_csv.parent / "episode_info.csv"))}
        lines: dict[str, list[str]] = {}
        for r in csv.DictReader(open(scripts_csv)):
            lines.setdefault(r["SEID"], []).append(f"{r['Character'].title()}: {r['Dialogue']}")
        docs = [Document(seid, info.get(seid, seid), "\n".join(ls)) for seid, ls in lines.items()]
    else:
        path = _fetch(SEINFELD_URL, data_dir / "seinfeld.parquet")
        for i, row in enumerate(pq.read_table(path).to_pylist()):
            docs.append(Document(f"ep{i:03d}", row["title"], row["script"]))

    by_title: dict[str, list[str]] = {}
    for d in docs:
        by_title.setdefault(_norm_title(d.title), []).append(d.doc_id)

    corpus = Corpus(docs)
    for title, summary in _wiki_summaries(data_dir).items():
        # Two-part episodes may be split into several scripts: all of them count as relevant.
        targets = by_title.get(title)
        if not targets:
            continue
        for j, sent in enumerate(re.split(r"(?<=[.!?])\s+", summary)):
            if len(sent.split()) >= 6:
                qid = f"{title}#{j}"
                corpus.queries[qid] = sent
                corpus.qrels[qid] = {t: 1 for t in targets}
    return corpus
