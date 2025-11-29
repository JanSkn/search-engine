import argparse
import gzip
import io
import pathlib
import sys
import requests
from tqdm import tqdm

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

OUTPUT_DIR = SCRIPT_DIR.parent / "index_builder" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = "msmarco.tsv.gz"

OUT_PATH = OUTPUT_DIR / OUTPUT_FILE


def open_stream(tsv_gz: str) -> io.BufferedReader:
    if tsv_gz.startswith(("http://", "https://")):
        resp = requests.get(tsv_gz, stream=True, timeout=60)
        resp.raise_for_status()
        raw = resp.raw
        if tsv_gz.endswith(".gz"):
            return gzip.GzipFile(fileobj=raw, mode="rb")
        else:
            return raw
    else:
        if tsv_gz.endswith(".gz"):
            return gzip.open(tsv_gz, "rb")
        else:
            return open(tsv_gz, "rb")


def convert_to_gzip(
    in_stream: io.BufferedReader,
    out_path: str,
    max_lines: int = None,
    chunk_size: int = 1024 * 1024,
) -> int:
    count = 0

    with gzip.open(out_path, "wb", compresslevel=6) as f_out:
        buffer = b""

        with tqdm(desc="Converting", unit="lines", mininterval=0.5) as pbar:
            while True:
                # Read in chunks for better performance
                chunk = in_stream.read(chunk_size)
                if not chunk:
                    # Process remaining buffer
                    if buffer:
                        lines = buffer.split(b"\n")
                        for line in lines:
                            if line:
                                f_out.write(line + b"\n")
                                count += 1
                                pbar.update(1)
                                if max_lines is not None and count >= max_lines:
                                    return count
                    break

                buffer += chunk
                lines = buffer.split(b"\n")

                # keep last incomplete line in buffer
                buffer = lines[-1]

                # process complete lines
                for line in lines[:-1]:
                    f_out.write(line + b"\n")
                    count += 1
                    pbar.update(1)

                    if max_lines is not None and count >= max_lines:
                        return count

    return count


def main():
    ap = argparse.ArgumentParser(
        description="Convert TSV to gzipped TSV (fast streaming with chunked reading)."
    )
    ap.add_argument("--tsv", required=True, help="Path or URL to TSV file")
    ap.add_argument(
        "--out",
        default=OUT_PATH,
        help=f"Output path (gzipped TSV). Default: {OUT_PATH}",
    )
    ap.add_argument(
        "--n", type=int, default=None, help="Number of lines to process (default: all)"
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Read chunk size in bytes (default: 1MB)",
    )
    args = ap.parse_args()

    print(f"Opening source: {args.tsv}", file=sys.stderr)
    if args.n:
        print(f"Processing first {args.n:,} lines", file=sys.stderr)
    else:
        print("Processing all lines", file=sys.stderr)

    with open_stream(args.tsv) as in_stream:
        count = convert_to_gzip(
            in_stream, args.out, max_lines=args.n, chunk_size=args.chunk_size
        )

    print(f"Done. Wrote {count:,} lines to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
