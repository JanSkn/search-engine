from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

from tqdm import tqdm

from .config import DatasetPaths, BuildConfig
from .io_utils import (
    iter_queries,
    sample_qids_from_qrels,
    load_qrels_for_qids,
    load_top100_selection,
)
from .features import parse_query_terms, build_postings_by_term, compute_features_for_doc

# Your inverted index class from C++ bindings
from cpp_utils import InvertedIndex  # type: ignore


def split_qids(qids: list[int], *, seed: int) -> tuple[set[int], set[int], set[int]]:
    rnd = random.Random(seed)
    qids = qids[:]
    rnd.shuffle(qids)

    n = len(qids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = set(qids[:n_train])
    val = set(qids[n_train : n_train + n_val])
    test = set(qids[n_train + n_val :])

    return train, val, test


def pick_negatives(
    *,
    pos_doc: int,
    hard_list: list[int],
    soft_candidates: dict[int, int],
    hard_n: int,
    soft_n: int,
    soft_rank: int,
    soft_fallback_from: int,
    soft_fallback_to: int,
) -> list[int]:
    # remove pos from candidates + dedupe while preserving order
    hard = []
    seen = {pos_doc}
    for d in hard_list:
        if d in seen:
            continue
        hard.append(d)
        seen.add(d)
        if len(hard) >= hard_n:
            break

    # soft: try exact rank first, else fallback from bottom (e.g. 100,99,...,80)
    soft: list[int] = []
    if soft_n > 0:
        ranks = []
        if soft_rank is not None:
            ranks.append(soft_rank)
        lo = min(soft_fallback_from, soft_fallback_to)
        hi = max(soft_fallback_from, soft_fallback_to)
        # go from hi..lo (bottom-up)
        ranks.extend(list(range(hi, lo - 1, -1)))

        for r in ranks:
            d = soft_candidates.get(r)
            if d is None or d in seen:
                continue
            soft.append(d)
            seen.add(d)
            if len(soft) >= soft_n:
                break

    return hard + soft


def build_one_example(
    inverted_index,
    *,
    qid: int,
    query: str,
    pos_doc: int,
    neg_docs: list[int],
) -> dict:
    query_terms = parse_query_terms(query)
    postings_by_term = build_postings_by_term(inverted_index, query_terms)

    docs = []

    # positive
    fv_pos = compute_features_for_doc(
        inverted_index,
        doc_id=pos_doc,
        query_terms=query_terms,
        postings_by_term=postings_by_term,
    )
    docs.append(
        {
            "doc_id": int(pos_doc),
            "label": 1,
            "features": fv_pos.as_dict(),
        }
    )

    # negatives
    for d in neg_docs:
        fv = compute_features_for_doc(
            inverted_index,
            doc_id=int(d),
            query_terms=query_terms,
            postings_by_term=postings_by_term,
        )
        docs.append(
            {
                "doc_id": int(d),
                "label": 0,
                "features": fv.as_dict(),
            }
        )

    return {"qid": int(qid), "query": query, "docs": docs}


def write_jsonl(path: Path, rows, *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if pretty:
                f.write(json.dumps(r, ensure_ascii=False))
            else:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Folder containing doctrain-*.tsv")
    ap.add_argument("--out-dir", type=str, required=True, help="Output folder for train/val/test jsonl")
    ap.add_argument("--index-dir", type=str, required=True, help="Path to your built inverted index base dir")

    ap.add_argument("--seed", type=int, default=BuildConfig.seed)
    ap.add_argument("--max-queries", type=int, default=100, help="Limit number of qids")

    ap.add_argument("--hard-negatives", type=int, default=BuildConfig.hard_negatives)
    ap.add_argument("--soft-negatives", type=int, default=BuildConfig.soft_negatives)
    ap.add_argument("--soft-rank", type=int, default=BuildConfig.soft_rank)
    ap.add_argument("--soft-fallback-from", type=int, default=BuildConfig.soft_fallback_from)
    ap.add_argument("--soft-fallback-to", type=int, default=BuildConfig.soft_fallback_to)

    ap.add_argument("--pretty-json", action="store_true")

    args = ap.parse_args()

    cfg = BuildConfig(
        seed=args.seed,
        max_queries=None if args.max_queries == -1 else int(args.max_queries),
        hard_negatives=int(args.hard_negatives),
        soft_negatives=int(args.soft_negatives),
        soft_rank=int(args.soft_rank),
        soft_fallback_from=int(args.soft_fallback_from),
        soft_fallback_to=int(args.soft_fallback_to),
        pretty_json=bool(args.pretty_json),
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    paths = DatasetPaths.from_base(data_dir, out_dir)

    # 1) sample qids from QRELS (guaranteed to have positives)
    qid_list = sample_qids_from_qrels(paths.qrels_tsv, seed=cfg.seed, max_queries=cfg.max_queries)
    qids = set(qid_list)

    # 1b) build qid->query map ONLY for those qids (streaming scan over queries.tsv)
    qid_to_query: dict[int, str] = {}
    missing = set(qids)

    for q in iter_queries(paths.queries_tsv):
        if q.qid in missing:
            qid_to_query[q.qid] = q.query
            missing.remove(q.qid)
            if not missing:
                break

    # drop qids that somehow have no query entry
    if missing:
        qids = set(qid_to_query.keys())
        qid_list = list(qids)

    # 2) split by qid (avoid leakage across query)
    train_qids, val_qids, test_qids = split_qids(qid_list, seed=cfg.seed)

    # 3) load qrels only for selected qids
    qrels = load_qrels_for_qids(paths.qrels_tsv, qids)

    # 4) load only needed top100 parts for selected qids
    top100_sel = load_top100_selection(
        paths.top100_tsv,
        qids,
        hard_k=cfg.hard_negatives,
        soft_rank=cfg.soft_rank,
        soft_fallback_from=cfg.soft_fallback_from,
        soft_fallback_to=cfg.soft_fallback_to,
    )

    # 5) open index once
    inverted_index = InvertedIndex(str(args.index_dir))

    # 6) streaming build: write train/val/test incrementally (RAM-efficient)
    out_dir.mkdir(parents=True, exist_ok=True)
    f_train = paths.train_jsonl.open("w", encoding="utf-8")
    f_val = paths.val_jsonl.open("w", encoding="utf-8")
    f_test = paths.test_jsonl.open("w", encoding="utf-8")
    ##########DEBUG
    sk_no_qrels = 0
    sk_no_top100 = 0
    sk_no_query = 0
    written = 0

    try:
        for qid in tqdm(qid_list, desc="Building LTR dataset", unit="qid"):
            query = qid_to_query.get(qid)
            if query is None:
                continue

            pos_doc = qrels.get(qid)
            if pos_doc is None:
                # no judged positive -> skip
                continue

            sel = top100_sel.get(qid)
            if sel is None:
                continue

            neg_docs = pick_negatives(
                pos_doc=pos_doc,
                hard_list=sel.hard,
                soft_candidates=sel.soft_candidates,
                hard_n=cfg.hard_negatives,
                soft_n=cfg.soft_negatives,
                soft_rank=cfg.soft_rank,
                soft_fallback_from=cfg.soft_fallback_from,
                soft_fallback_to=cfg.soft_fallback_to,
            )
            ##########DEBUG
            query = qid_to_query.get(qid)
            if query is None:
                sk_no_query += 1
                continue

            pos_doc = qrels.get(qid)
            if pos_doc is None:
                sk_no_qrels += 1
                continue

            sel = top100_sel.get(qid)
            if sel is None:
                sk_no_top100 += 1
                continue

            # ensure final count = 1 + hard + soft
            # (If data quality issues cause fewer, we still write what we have.)
            row = build_one_example(
                inverted_index,
                qid=qid,
                query=query,
                pos_doc=pos_doc,
                neg_docs=neg_docs,
            )
            #######DEBUG
            written += 1
            s = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            if cfg.pretty_json:
                s = json.dumps(row, ensure_ascii=False)

            if qid in train_qids:
                f_train.write(s + "\n")
            elif qid in val_qids:
                f_val.write(s + "\n")
            else:
                f_test.write(s + "\n")

            del row, s

            if written % 500 == 0:
                gc.collect()

    finally:
        f_train.close()
        f_val.close()
        f_test.close()

    print("SUMMARY")
    print("  qids_total:", len(qid_list))
    print("  written:", written)
    print("  skipped_no_query:", sk_no_query)
    print("  skipped_no_qrels:", sk_no_qrels)
    print("  skipped_no_top100:", sk_no_top100)
    print("  qrels_loaded:", len(qrels))
    print("  top100_loaded:", len(top100_sel))

    print(f"Wrote:\n  {paths.train_jsonl}\n  {paths.val_jsonl}\n  {paths.test_jsonl}")


if __name__ == "__main__":
    main()