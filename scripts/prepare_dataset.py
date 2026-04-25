from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = 42

ONESTOP_JSONL = RAW_DIR / "onestop_parsed.jsonl"
CLEAR_JSONL = RAW_DIR / "clear_parsed.jsonl"

TRAIN_JSONL = PROCESSED_DIR / "train.jsonl"
VAL_JSONL = PROCESSED_DIR / "val.jsonl"
TEST_JSONL = PROCESSED_DIR / "test.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dedup(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for r in records:
        key = r["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratify by (domain, load_label) for balanced splits."""
    # Group by stratum
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = f"{r['domain']}_{r['load_label']}"
        strata[key].append(r)

    train_all, val_all, test_all = [], [], []
    for key, group in strata.items():
        random.seed(seed)
        random.shuffle(group)
        n = len(group)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * (1 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test
        if n_train < 1:
            # Too small — put everything in train
            train_all.extend(group)
            continue
        train_all.extend(group[:n_train])
        val_all.extend(group[n_train : n_train + n_val])
        test_all.extend(group[n_train + n_val :])

    return train_all, val_all, test_all


def print_stats(name: str, records: list[dict]) -> None:
    df = pd.DataFrame(records)
    print(f"\n{name}: {len(records):,} samples")
    if "domain" in df.columns:
        print(df.groupby(["domain", "load_label"]).size().to_string())


def main() -> None:
    print("Loading raw data …")
    onestop = load_jsonl(ONESTOP_JSONL)
    clear = load_jsonl(CLEAR_JSONL)

    if not onestop:
        print("WARNING: No OneStopEnglish data found. Run scripts/download_data.py first.")

    all_records = onestop + clear
    print(f"Total before dedup: {len(all_records):,}")
    all_records = dedup(all_records)
    print(f"Total after dedup:  {len(all_records):,}")

    if not all_records:
        print("ERROR: No records to split. Aborting.")
        return

    # Ensure required fields
    for r in all_records:
        if "load_label" not in r:
            s = r.get("load_score", 50)
            r["load_label"] = "Low" if s < 40 else ("Medium" if s <= 70 else "High")
        if "domain" not in r:
            r["domain"] = "news"
        if "source" not in r:
            r["source"] = "unknown"

    train, val, test = stratified_split(all_records)

    print_stats("Train", train)
    print_stats("Val", val)
    print_stats("Test", test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSaving splits …")
    save_jsonl(train, TRAIN_JSONL)
    save_jsonl(val, VAL_JSONL)
    save_jsonl(test, TEST_JSONL)

    print("\nDataset preparation complete.")
    print(f"Expected 10,000–13,000 samples total with both datasets.")
    print(f"Got {len(all_records):,} samples. (Add CLEAR corpus for more.)")


if __name__ == "__main__":
    main()
