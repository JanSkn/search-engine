import requests
import gzip
import shutil
from pathlib import Path


def download_and_extract_msmarco():
    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"
    SCRIPT_DIR = Path(__file__).resolve().parent

    OUTPUT_DIR = SCRIPT_DIR.parent / "index_builder" / "data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gz_file = OUTPUT_DIR / "msmarco-docs.tsv.gz"
    tsv_file = OUTPUT_DIR / "msmarco-docs.tsv"

    print(f"Downloading {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(gz_file, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end="")

    print(f"Unzipping to {tsv_file}...")
    with gzip.open(gz_file, "rb") as f_in:
        with open(tsv_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("Unzipping completed")

    gz_file.unlink()

    return tsv_file


if __name__ == "__main__":
    result = download_and_extract_msmarco()
    print(f"\nFile saved at: {result}")
