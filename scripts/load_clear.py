from __future__ import annotations

import json
import warnings
from pathlib import Path

CLEAR_CSV = Path("data/raw/clear.csv")
OUTPUT_JSONL = Path("data/raw/clear_parsed.jsonl")


def score_to_label(score: int | float) -> str:
    if score < 40:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


def infer_domain(text: str, source_col: str = "") -> str:
    text_lower = (text + source_col).lower()
    if any(w in text_lower for w in ["law", "court", "statute", "legal", "regulation", "act ", "section ", "pursuant"]):
        return "legal"
    if any(w in text_lower for w in ["patient", "clinical", "medical", "hospital", "diagnosis", "treatment", "disease", "drug", "dosage"]):
        return "medical"
    return "legal"  # default for CLEAR which is primarily legal/medical


def parse_clear_csv(csv_path: Path) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect text column
    text_col = None
    for candidate in ["text", "sentence", "content", "passage", "document"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # Try first string column
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                break
    if text_col is None:
        raise ValueError(f"Cannot find text column in CSV. Columns: {list(df.columns)}")

    # Detect complexity / label column
    complexity_col = None
    for candidate in ["complexity", "label", "level", "type", "class", "simplicity"]:
        if candidate in df.columns:
            complexity_col = candidate
            break

    records: list[dict] = []
    for idx, row in df.iterrows():
        text = str(row[text_col]).strip()
        if len(text) < 20:
            continue

        # Determine complexity
        if complexity_col and pd.notna(row[complexity_col]):
            val = str(row[complexity_col]).strip().lower()
            if val in ("plain", "simple", "simplified", "0", "easy"):
                complexity = "plain"
                load_score = 15
            else:
                complexity = "complex"
                load_score = 80
        else:
            # Fallback: heuristic based on avg word length
            words = text.split()
            avg_len = sum(len(w) for w in words) / max(len(words), 1)
            if avg_len > 6.5:
                complexity = "complex"
                load_score = 80
            else:
                complexity = "plain"
                load_score = 15

        domain = infer_domain(text, str(row.get("source", "")))

        records.append(
            {
                "text": text,
                "load_score": load_score,
                "load_label": score_to_label(load_score),
                "domain": domain,
                "source": "clear",
                "article_id": f"clear_{idx:05d}",
                "complexity": complexity,
            }
        )

    return records


def main() -> None:
    if not CLEAR_CSV.exists():
        warnings.warn(
            f"CLEAR corpus CSV not found at {CLEAR_CSV}. "
            "Fill the form at https://github.com/lauramanor/clear-corpus "
            "and place the downloaded file at data/raw/clear.csv. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return

    print(f"Loading CLEAR corpus from {CLEAR_CSV} …")
    records = parse_clear_csv(CLEAR_CSV)

    if not records:
        print("WARNING: No valid records parsed from CLEAR corpus.")
        return

    print(f"Parsed {len(records)} CLEAR records.")
    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["load_label"]] = label_counts.get(r["load_label"], 0) + 1
    print(f"Label distribution: {label_counts}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records → {OUTPUT_JSONL}")
    print("Done. Run scripts/prepare_dataset.py to create train/val/test splits.")


if __name__ == "__main__":
    main()
