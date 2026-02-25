from __future__ import annotations

import sys
import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Iterable

import numpy as np
from backend.logging_config import get_logger
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.search_engine.index.index_loader import get_index
from backend.search_engine.scoring.bm25 import bm25_score_docs, BM25Config
from cpp_utils import PostingList  # type: ignore
from functools import lru_cache


project_root = "/home/vietc/projects/search-engine/src"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger = get_logger(__name__)
inverted_index = get_index()
metadata = inverted_index.metadata


@lru_cache(maxsize=10_000)
def get_document_text(doc_id: int) -> str | None:
    """
    get full text from doc id
    """
    # get offset for doc id in msmarco file
    offset = inverted_index.doc_store.get_tsv_offset(doc_id)
    if offset is None:
        logger.error(f"no offset for doc {doc_id} found")
        return None
    
    tsv_path = "/home/vietc/projects/search-engine/src/backend/search_engine/index_builder/data/msmarco-docs.tsv"  # Hier den richtigen Pfad einfügen
    
    try:
        with open(tsv_path, 'rb') as f:
            f.seek(offset)
            
            line = f.readline()
            
            # parse tsv line (doc_id \t url \t title \t text)
            parts = line.decode("utf-8", errors="replace").rstrip("\n").split("\t")
            
            if len(parts) >= 4:
                # doc_id, url, title, text
                doc_id_from_file = parts[0]
                url = parts[1]
                title = parts[2]
                full_text = parts[3]
                
                logger.debug(f"doc {doc_id} loaded: {title}")
                return full_text
            else:
                logger.error(f"unknown tsv format: {len(parts)} colomnus")
                return None
                
    except FileNotFoundError:
        logger.error(f"tsv file not found: {tsv_path}")
        return None
    except Exception as e:
        logger.error(f"error reading doc {doc_id}: {e}")
        return None


class SimpleTokenizer:
    _re = re.compile(r"[A-Za-z0-9]+")

    def tokenize(self, text: str) -> list[str]:
        return [m.group(0).lower() for m in self._re.finditer(text)]


# TF-IDF (fit once on docs)


@dataclass
class TfidfModel:
    vectorizer: TfidfVectorizer

    def encode(self, texts: list[str]):
        return self.vectorizer.transform(texts)

def fit_tfidf_on_corpus(docs_iter: Iterable[str], max_features: int = 200_000) -> TfidfModel:
    """
    fit tf-idf once on a corpus of documents (or a large sampel)
    """
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        max_features=max_features,
        ngram_range=(1, 1),
    )
    vec.fit(docs_iter)
    return TfidfModel(vectorizer=vec)

def cosine_sparse(a, b) -> float:
    # a,b are 1xV sparse
    denom = (np.sqrt(a.multiply(a).sum()) * np.sqrt(b.multiply(b).sum()))
    if denom == 0.0:
        return 0.0
    return float(a.multiply(b).sum() / denom)


# faeture computation

def compute_features_for_pair(
    *,
    qid: str,
    query: str,
    docid: int,
    doc_text: str,
    tokenizer: SimpleTokenizer,
    tfidf: TfidfModel,
    bm25_cfg: BM25Config = BM25Config(),
) -> dict[str, float]:
    q_terms = tokenizer.tokenize(query)
    d_terms = tokenizer.tokenize(doc_text)

    q_set = set(q_terms)
    d_set = set(d_terms)

    matched = sum(1 for t in q_set if t in d_set)
    frac = matched / max(1, len(q_set))

    # exact phrase match (case-insensitive, normalize whitespace)
    q_norm = " ".join(q_terms).strip()
    d_norm = " ".join(d_terms).strip()
    phrase = 1.0 if (q_norm and q_norm in d_norm) else 0.0

    # tf-idf cosine
    q_vec = tfidf.encode([query])
    d_vec = tfidf.encode([doc_text])
    tfidf_cos = cosine_sparse(q_vec, d_vec)

    # bm25 score for whole document (single candidate doc)
    postings = {}
    for t in q_terms:
        pl = inverted_index.index.get(t)
        if pl is not None:
            postings[t] = pl
    bm25_scores = bm25_score_docs(
        query_terms=q_terms,
        postings_by_term=postings,
        candidate_doc_ids=[docid],
        num_docs=metadata.num_docs,
        avgdl=metadata.avg_doc_length,
        get_doc_length=metadata.get_doc_length,
        cfg=bm25_cfg,
    )
    bm25 = float(bm25_scores.get(docid, 0.0))

    return {
        "bm25": bm25,
        "tfidf_cos": float(tfidf_cos),
        "matched_terms": float(matched),
        "matched_frac": float(frac),
        "phrase_match": float(phrase),
    }
