import itertools
import time

import torch
from backend.search_engine.index_builder.create_embeddings import (
    TSV_PATH,
    MSMarcoDataset,
    NumpyIndexer,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

MAX_DOCS = 5000
WORKER_OPTIONS = [1, 2, 4, 8]
BATCH_SIZES = [16, 32, 64, 128]

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_built()
    else torch.device("cpu")
)


def benchmark(num_workers: int, batch_size: int):
    indexer = NumpyIndexer(
        num_workers=num_workers,
        batch_size=batch_size,
        max_docs=MAX_DOCS,
    )

    dataset = MSMarcoDataset(
        TSV_PATH,
        indexer.embedder.tokenizer,
        MAX_DOCS,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if DEVICE == "cuda" else False,
        shuffle=False,
    )

    embedder = indexer.embedder
    embedder.model.eval()

    total_docs = 0
    start = time.perf_counter()

    with torch.no_grad():
        for docids, encoded_batch in tqdm(
            dataloader,
            desc=f"workers={num_workers} bs={batch_size}",
            leave=False,
        ):
            encoded_batch = {
                k: v.to(embedder.model.device) for k, v in encoded_batch.items()
            }

            embedder.embed_tokenized(encoded_batch)
            total_docs += len(docids)

    elapsed = time.perf_counter() - start
    docs_per_sec = total_docs / elapsed

    return elapsed, docs_per_sec


def main():
    results = []

    for num_workers, batch_size in itertools.product(WORKER_OPTIONS, BATCH_SIZES):
        print(f"\nTesting workers={num_workers}, batch_size={batch_size}")
        elapsed, throughput = benchmark(num_workers, batch_size)

        print(f"→ Time: {elapsed:.2f}s | Throughput: {throughput:.2f} docs/sec")

        results.append(
            {
                "workers": num_workers,
                "batch_size": batch_size,
                "time": elapsed,
                "throughput": throughput,
            }
        )

    results.sort(key=lambda x: x["throughput"], reverse=True)

    best = results[0]

    print("\n" + "=" * 50)
    print("🏆 BEST CONFIGURATION")
    print("=" * 50)
    print(
        f"Workers: {best['workers']}\n"
        f"Batch Size: {best['batch_size']}\n"
        f"Throughput: {best['throughput']:.2f} docs/sec\n"
        f"Time: {best['time']:.2f}s"
    )


if __name__ == "__main__":
    main()
