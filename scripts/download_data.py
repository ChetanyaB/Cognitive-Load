from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path("data/raw")
ONESTOP_DIR = RAW_DIR / "onestop"
ONESTOP_ZIP_URL = (
    "https://github.com/nishkalavallabhi/OneStopEnglishCorpus/archive/refs/heads/master.zip"
)
ONESTOP_ZIP_PATH = RAW_DIR / "onestop.zip"

LEVEL_TO_SCORE: dict[str, int] = {
    "ele": 20,
    "int": 55,
    "adv": 85,
}

LEVEL_ALIASES: dict[str, str] = {
    "ele": "ele",
    "elementary": "ele",
    "int": "int",
    "intermediate": "int",
    "adv": "adv",
    "advanced": "adv",
}


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} …")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"Saved to {dest}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting {zip_path} …")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Extraction complete.")


def detect_level(filename: str) -> str | None:
    lower = filename.lower()
    for alias, canonical in LEVEL_ALIASES.items():
        if alias in lower:
            return canonical
    return None


def parse_onestop(onestop_root: Path) -> list[dict]:
    records: list[dict] = []
    article_id = 0

    # Walk all text files inside the extracted archive
    for path in sorted(onestop_root.rglob("*.txt")):
        level = detect_level(path.name) or detect_level(str(path.parent))
        if level is None:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if len(text) < 50:
            continue
        score = LEVEL_TO_SCORE[level]
        records.append(
            {
                "text": text,
                "load_score": score,
                "load_label": score_to_label(score),
                "domain": "news",
                "source": "onestop",
                "article_id": f"onestop_{article_id:05d}",
            }
        )
        article_id += 1

    return records


def score_to_label(score: int | float) -> str:
    if score < 40:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records → {path}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download if not already present
    if not ONESTOP_ZIP_PATH.exists():
        download_file(ONESTOP_ZIP_URL, ONESTOP_ZIP_PATH)
    else:
        print(f"Zip already exists at {ONESTOP_ZIP_PATH}, skipping download.")

    # Extract
    if not ONESTOP_DIR.exists():
        extract_zip(ONESTOP_ZIP_PATH, ONESTOP_DIR)
    else:
        print(f"Already extracted at {ONESTOP_DIR}, skipping extraction.")

    # Parse
    records = parse_onestop(ONESTOP_DIR)
    if not records:
        print("WARNING: No records parsed. Check extraction path.")
        return

    print(f"Parsed {len(records)} OneStopEnglish articles.")
    level_counts: dict[str, int] = {}
    for r in records:
        level_counts[r["load_label"]] = level_counts.get(r["load_label"], 0) + 1
    print(f"Label distribution: {level_counts}")

    save_jsonl(records, RAW_DIR / "onestop_parsed.jsonl")
    print("Done. Run scripts/prepare_dataset.py to create train/val/test splits.")


if __name__ == "__main__":
    main()
