import gzip
import json
import re
from typing import Iterable

import spacy
import numpy as np

from backend.search_engine.index.inverted_index import InvertedIndex
from backend.search_engine.models.index import PostingList
from backend.search_engine.query.query_preprocessing import AND, OR, NOT

KEEP_TOKENS = AND | OR | NOT | {"(", ")"}

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat", "senter"])


def lemmatize_search_query(query: str) -> list[str]:
    result = []

    doc = nlp(query)

    for token in doc:
        word = token.text

        if word in KEEP_TOKENS:
            result.append(word)
            continue

        lemma = token.lemma_.lower()
        result.append(lemma)

    return result


def _iter_jsonl(path: str) -> Iterable[tuple[str, str, str, str]]:
    # yields (docid, url, title, body) from jsonl or jsonl.gz
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docid = obj.get("doc_id", "")
            url = obj.get("url", "")
            title = obj.get("title", "")
            body = obj.get("body", "")
            yield docid, url, title, body


def _should_keep(tok) -> bool:
    # keep alphabetic words, digits, urls, emails, and hyphen words
    if tok.is_space or tok.is_punct:
        return False
    if tok.like_url or tok.like_email:
        return True
    if tok.is_alpha or tok.is_digit:
        return True
    if re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)+", tok.text.lower()):
        return True
    return False


def build_index(
    jsonl_path: str,
    limit: int | None = None,
    lang: str = "en",
    use_lemma: bool = True,
    batch_size: int = 100,
) -> InvertedIndex:
    inv = InvertedIndex()

    docs_batch = []
    doc_ids_batch = []
    metadata_batch = []

    for doc_id, (_, url, title, body) in enumerate(_iter_jsonl(jsonl_path)):
        if limit is not None and doc_id >= limit:
            break

        metadata_batch.append({"url": url, "title": title})
        text = f"{title}\n\n{body}".strip().lower()

        docs_batch.append(text)
        doc_ids_batch.append(doc_id)

        if len(docs_batch) >= batch_size:
            _process_batch(
                nlp, inv, docs_batch, doc_ids_batch, metadata_batch, use_lemma
            )
            docs_batch = []
            doc_ids_batch = []
            metadata_batch = []

    if docs_batch:
        _process_batch(nlp, inv, docs_batch, doc_ids_batch, metadata_batch, use_lemma)

    inv.finalize()
    return inv


def _process_batch(nlp, inv, texts, doc_ids, metadata_list, use_lemma):
    for doc_id, metadata, doc in zip(
        doc_ids, metadata_list, nlp.pipe(texts, batch_size=50)
    ):
        inv.doc_store[doc_id] = metadata

        for pos, tok in enumerate(doc):
            if not _should_keep(tok):
                continue

            # use either lemma or raw text
            term_text = tok.text.lower()
            if use_lemma and getattr(tok, "lemma_", None):
                lemma_lc = tok.lemma_.lower()
                term = lemma_lc if lemma_lc else term_text
            else:
                term = term_text

            # get existing posting list or make new
            pl = inv.index.get(term)
            if pl is None:
                # new unknown term
                pl = PostingList(
                    postings=np.array([doc_id]),
                    term_frequencies={doc_id: 1},
                    positions={doc_id: np.array([pos], dtype=int)},
                )
                inv.index[term] = pl
            else:
                if doc_id in pl.term_frequencies:
                    # update existing doc entry
                    pl.term_frequencies[doc_id] += 1
                    pl.positions[doc_id] = np.append(pl.positions[doc_id], pos)
                else:
                    # new doc for this term
                    pl.postings = np.append(pl.postings, doc_id)
                    pl.term_frequencies[doc_id] = 1
                    pl.positions[doc_id] = np.array([pos], dtype=int)
