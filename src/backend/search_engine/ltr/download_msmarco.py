from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

from .config import LTRPaths

URLS = {
    "msmarco-doctrain-queries.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz",
    "msmarco-doctrain-qrels.tsv.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz",
    "msmarco-doctrain-top100.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz",
}

def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)

def _gunzip(src: Path, dst: Path) -> None:
    print(f"Unzipping {src} -> {dst}")
    with gzip.open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

def main() -> None:
    paths = LTRPaths()
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    # queries
    q_gz = paths.data_dir / "msmarco-doctrain-queries.tsv.gz"
    _download(URLS[q_gz.name], q_gz)
    _gunzip(q_gz, paths.queries_tsv)

    # qrels
    r_gz = paths.data_dir / "msmarco-doctrain-qrels.tsv.gz"
    _download(URLS[r_gz.name], r_gz)
    _gunzip(r_gz, paths.qrels_tsv)

    # top100
    t_gz = paths.data_dir / "msmarco-doctrain-top100.gz"
    _download(URLS[t_gz.name], t_gz)
    _gunzip(t_gz, paths.top100_tsv)

    print("Done. note: docs collection is not included here")

if __name__ == "__main__":
    main()