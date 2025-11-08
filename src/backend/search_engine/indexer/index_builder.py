import gzip
import json
import re
from typing import Iterable

import spacy

from backend.search_engine.index.inverted_index import InvertedIndex
from backend.search_engine.models.index import PostingList
from backend.search_engine.query.query_preprocessing import AND, OR, NOT

KEEP_TOKENS = AND | OR | NOT | {"(", ")"}

# TODO can be different model in build_index
nlp = spacy.load("en_core_web_sm")


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
    spacy_model: str | None = "en_core_web_sm",
    use_lemma: bool = True,
) -> InvertedIndex:
    # loads docs from jsonl, tokenizes, and builds an inverted index with positions
    if spacy_model:
        nlp = spacy.load(spacy_model, disable=["ner", "parser", "textcat", "senter"])
    else:
        nlp = spacy.blank(lang)

    inv = InvertedIndex()

    for doc_id, (docid, url, title, body) in enumerate(_iter_jsonl(jsonl_path)):
        if limit is not None and doc_id >= limit:
            break
        
        inv.doc_store[doc_id] = url

        # build text to tokenize
        text = f"{title}\n\n{body}".strip().lower()
        doc = nlp(text)

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
                # new unkown term
                pl = PostingList(
                    urls=[url],
                    titles=[title],
                    doc_freq=1,
                    term_frequencies=[1],
                    positions=[[pos]],
                    postings=[doc_id],
                    skip_pointers={},
                )
                inv.index[term] = pl
            else:
                # term exists: check if last entry is already this document
                if pl.postings and pl.postings[-1] == doc_id:
                    pl.term_frequencies[-1] += 1
                    pl.positions[-1].append(pos)
                else:
                    # no previous entry in posting list for this document
                    pl.urls.append(url)
                    pl.titles.append(title)
                    pl.postings.append(doc_id)
                    pl.term_frequencies.append(1)
                    pl.positions.append([pos])
                    pl.doc_freq = len(pl.postings)

    inv.finalize()
    return inv
