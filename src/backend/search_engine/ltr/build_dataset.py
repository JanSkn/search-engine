from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .config import LTRPaths, SamplingConfig, TrainConfig, DatasetBuildConfig
from .utils_io import load_queries, load_qrels, iter_top100_for_qids, Top100Entry
from .features import compute_features_for_pair, get_document_text, fit_tfidf_on_corpus, SimpleTokenizer
from backend.logging_config import get_logger

logger = get_logger(__name__)

def split_qids(qids: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    rnd = random.Random(seed)
    qids = qids[:]
    rnd.shuffle(qids)

    n = len(qids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = qids[:n_train]
    val = qids[n_train:n_train + n_val]
    test = qids[n_train + n_val:]

    logger.info(f"Split results: train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


def sample_candidates_for_qid(
    *,
    qid: str,
    pos_docids: set[int],
    top100: list[Top100Entry],
    cfg: SamplingConfig,
    seed: int,
) -> tuple[int, list[int]]:
    """
    Returns: (pos_docid, negatives_docids)
    """
    rnd = random.Random(f"{seed}:{qid}")

    # pick 1 positive
    if not pos_docids:
        raise ValueError(f"qid {qid} has no positives in qrels")
    pos_docid = sorted(pos_docids)[0]  # deterministic
    # if pos_docid not in top100, still ok

    # hard negatives from top ranks
    hard: list[int] = []
    for e in top100:
        if len(hard) >= cfg.hard_top_k:
            break
        if e.docid in pos_docids:
            continue
        hard.append(e.docid)

    # easy negatives from bottom ranks range
    bottom = [e.docid for e in top100 if cfg.easy_from_rank_low <= e.rank <= cfg.easy_from_rank_high and e.docid not in pos_docids]
    rnd.shuffle(bottom)
    easy = bottom[: cfg.easy_n]

    negatives = hard + easy
    # de-dup
    seen = set()
    uniq: list[int] = []
    for d in negatives:
        if d not in seen:
            uniq.append(d)
            seen.add(d)

    logger.debug(f"qid {qid}: pos={pos_docid}, hard={len(hard)}, easy={len(easy)}, total_neg={len(uniq)}")
    return pos_docid, uniq


def load_top100_for_qids(top100_path: Path, split_qids: list[str]) -> dict[str, list[Top100Entry]]:
    qid_set = set(split_qids)
    out: dict[str, list[Top100Entry]] = {}
    for qid, entry in iter_top100_for_qids(top100_path, qid_set):
        out.setdefault(qid, []).append(entry)
    # sort by rank
    for qid in out:
        out[qid].sort(key=lambda e: e.rank)
    return out

def iter_train_texts(docids):
    for d in docids:
        txt = get_document_text(d)
        if txt:
            yield txt


def write_jsonl(path: Path, rows: Iterable[dict], mode: str = "w") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    return count



def build_split_jsonl(
    *,
    split_qids: list[str],
    queries: dict[str, str],
    qrels: dict[str, set[int]],
    top100_by_qid: dict[str, list[Top100Entry]],
    tfidf_model,
    tokenizer: SimpleTokenizer,
    out_path: Path,
    samp_cfg: SamplingConfig,
    seed: int,
    list_size: int, 
    mode: str = "w"
) -> None:
    start_time = time.time()
    last_log_time = start_time
    logger.info(f"Building {out_path.name} with {len(split_qids)} QIDs")
    skipped_no_query = 0
    skipped_no_pos = 0
    skipped_no_top100 = 0
    skipped_missing_text = 0
    skipped_too_small = 0
    processed = 0
    
    def _rows():
        nonlocal skipped_no_query, skipped_no_pos, skipped_no_top100, skipped_missing_text, skipped_too_small, processed, start_time, last_log_time
        
        for qid in split_qids:
            query = queries.get(qid)
            if not query:
                skipped_no_query += 1
                continue
            pos = qrels.get(qid, set())
            if not pos:
                skipped_no_pos += 1
                continue
            top100 = top100_by_qid.get(qid, [])
            if not top100:
                skipped_no_top100 += 1
                continue

            pos_docid, negs = sample_candidates_for_qid(
                qid=qid,
                pos_docids=pos,
                top100=top100,
                cfg=samp_cfg,
                seed=seed,
            )

            # enforce fixed list size
            docs = [pos_docid] + negs
            docs = docs[:list_size]
            if len(docs) < list_size:
                # pad with additional negatives from top100 (skipping positives) if needed
                extra = [e.docid for e in top100 if e.docid not in pos]
                for d in extra:
                    if d not in docs:
                        docs.append(d)
                    if len(docs) >= list_size:
                        break
            if len(docs) < list_size:
                skipped_too_small += 1
                continue

            labels = [1.0] + [0.0] * (list_size - 1)

            feat_list = []
            ok = True
            for docid in docs:
                text = get_document_text(docid)
                if not text:
                    logger.debug(f"Missing text for docid {docid} in qid {qid}")
                    ok = False
                    break
                feats = compute_features_for_pair(
                    qid=qid,
                    query=query,
                    docid=docid,
                    doc_text=text,
                    tokenizer=tokenizer,
                    tfidf=tfidf_model,
                )
                feat_list.append(feats)

            if not ok:
                skipped_missing_text += 1
                continue  # skip this qid entirely
            
            processed += 1
            if processed % 100 == 0:
                now = time.time()
                elapsed = now - start_time
                delta = now - last_log_time
                last_log_time = now

                qps = 100 / delta if delta > 0 else 0

                total_qids = len(split_qids)
                remaining = total_qids - processed

                if qps > 0:
                    eta_seconds = remaining / qps
                    eta_min = eta_seconds / 60
                else:
                    eta_min = 0

                logger.info(
                    f"Processed {processed}/{total_qids} QIDs "
                    f"| elapsed={elapsed:.1f}s "
                    f"| last100={delta:.1f}s "
                    f"| speed={qps:.2f} QIDs/s "
                    f"| ETA ≈ {eta_min:.1f} min"
                )

            yield {
                "qid": qid,
                "query": query,
                "docids": docs,
                "labels": labels,
                "features": feat_list,
            }
    
    written = write_jsonl(out_path, _rows(), mode=mode)

    total_time = time.time() - start_time
    logger.info(
        f"{out_path.name} finished in {total_time:.1f}s "
        f"({total_time/60:.2f} min)"
    )

    logger.info(f"Written {written} rows to {out_path} (mode={mode})")
    
    logger.info(f"Statistics for {out_path.name}:")
    logger.info(f"  - Processed: {processed}")
    logger.info(f"  - Skipped (no query): {skipped_no_query}")
    logger.info(f"  - Skipped (no positives): {skipped_no_pos}")
    logger.info(f"  - Skipped (no top100): {skipped_no_top100}")
    logger.info(f"  - Skipped (missing text): {skipped_missing_text}")
    logger.info(f"  - Skipped (too small list): {skipped_too_small}")

def main() -> None:
    logger.info("Starting LTR data preparation")

    paths = LTRPaths()
    tcfg = TrainConfig()
    scfg = SamplingConfig()
    bcfg = DatasetBuildConfig()

    logger.info(f"Loading data from: {paths.queries_tsv}, {paths.qrels_tsv}, {paths.top100_tsv}")

    # load only small meta files
    queries = load_queries(paths.queries_tsv)
    qrels = load_qrels(paths.qrels_tsv)

    logger.info(f"Loaded {len(queries)} queries, {len(qrels)} qrels")

    # build candidate QIDs w/o loading top100 globally
    qids = sorted(set(queries.keys()) & set(qrels.keys()))
    logger.info(f"Found {len(qids)} QIDs that have query + qrels")

    train_qids, val_qids, test_qids = split_qids(qids, seed=tcfg.seed)

    # smoke test limiting (only affects runtime/size; same logic works for full run)
    if bcfg.smoke_limit_qids is not None:
        train_qids = train_qids[: bcfg.smoke_limit_qids]
        val_qids = val_qids[: max(1, bcfg.smoke_limit_qids // 5)]
        test_qids = test_qids[: max(1, bcfg.smoke_limit_qids // 5)]
        logger.info(
            f"SMOKE LIMIT active: train={len(train_qids)}, val={len(val_qids)}, test={len(test_qids)}"
        )

    tokenizer = SimpleTokenizer()

    # ----------------------------
    # 1) Fit TF-IDF model (small sample of train QIDs)
    # ----------------------------
    logger.info("Sampling documents for TF-IDF fitting")

    tfidf_qids = train_qids[: min(len(train_qids), bcfg.tfidf_fit_qids)]
    logger.info(f"TF-IDF will use top100 docs from {len(tfidf_qids)} train QIDs")

    top100_tfidf = load_top100_for_qids(paths.top100_tsv, tfidf_qids)

    train_docids_sample: list[int] = []
    for qid in tfidf_qids:
        for e in top100_tfidf.get(qid, [])[:100]:
            train_docids_sample.append(e.docid)

    unique_docids = list(dict.fromkeys(train_docids_sample).keys())
    logger.info(f"Collected {len(unique_docids)} unique docids for TF-IDF fit")

    tfidf_model = fit_tfidf_on_corpus(iter_train_texts(unique_docids), max_features=bcfg.max_features)
    logger.info("TF-IDF model fitted successfully")

    del top100_tfidf

    # ----------------------------
    # 2) Build VAL/TEST (load top100 only for those QIDs)
    # ----------------------------
    logger.info("Building validation split")
    top100_val = load_top100_for_qids(paths.top100_tsv, val_qids)
    val_qids = [q for q in val_qids if q in top100_val]
    build_split_jsonl(
        split_qids=val_qids,
        queries=queries,
        qrels=qrels,
        top100_by_qid=top100_val,
        tfidf_model=tfidf_model,
        tokenizer=tokenizer,
        out_path=paths.val_jsonl,
        samp_cfg=scfg,
        seed=tcfg.seed,
        list_size=tcfg.list_size,
        mode="w",
    )
    del top100_val

    logger.info("Building test split")
    top100_test = load_top100_for_qids(paths.top100_tsv, test_qids)
    test_qids = [q for q in test_qids if q in top100_test]
    build_split_jsonl(
        split_qids=test_qids,
        queries=queries,
        qrels=qrels,
        top100_by_qid=top100_test,
        tfidf_model=tfidf_model,
        tokenizer=tokenizer,
        out_path=paths.test_jsonl,
        samp_cfg=scfg,
        seed=tcfg.seed,
        list_size=tcfg.list_size,
        mode="w",
    )
    del top100_test

    # ----------------------------
    # 3) Build TRAIN chunked (append to one file)
    # ----------------------------
    logger.info("Building training split (chunked)")

    # reset train output file
    paths.train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    open(paths.train_jsonl, "w", encoding="utf-8").close()

    # cchunk size (keep safe for low ram)
    chunk_size = 1_000 if bcfg.smoke_limit_qids is None else min(1_000, len(train_qids))
    logger.info(f"Train chunk_size={chunk_size}")

    for start in range(0, len(train_qids), chunk_size):
        chunk_start_time = time.time()
        chunk_qids = train_qids[start : start + chunk_size]
        logger.info(f"Train chunk {start}..{start + len(chunk_qids)} / {len(train_qids)}")

        top100_chunk = load_top100_for_qids(paths.top100_tsv, chunk_qids)
        chunk_qids = [q for q in chunk_qids if q in top100_chunk]

        build_split_jsonl(
            split_qids=chunk_qids,
            queries=queries,
            qrels=qrels,
            top100_by_qid=top100_chunk,
            tfidf_model=tfidf_model,
            tokenizer=tokenizer,
            out_path=paths.train_jsonl,
            samp_cfg=scfg,
            seed=tcfg.seed,
            list_size=tcfg.list_size,
            mode="a",  # append
        )
    
        chunk_time = time.time() - chunk_start_time
        logger.info(
            f"Chunk {start}..{start+len(chunk_qids)} "
            f"finished in {chunk_time:.1f}s"
        )

        del top100_chunk

    logger.info("LTR data preparation completed successfully")


if __name__ == "__main__":
    main()