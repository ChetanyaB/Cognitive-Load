# Data Directory

## Structure

```
data/
├── raw/                    # Raw downloaded or user-supplied data
│   ├── onestop/            # Extracted OneStopEnglish corpus
│   ├── onestop.zip         # Downloaded zip (can delete after extraction)
│   ├── onestop_parsed.jsonl  # Parsed OneStopEnglish records
│   ├── clear.csv           # User-supplied CLEAR corpus CSV (see below)
│   └── clear_parsed.jsonl  # Parsed CLEAR corpus records
│
└── processed/              # Train/val/test splits (created by prepare_dataset.py)
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

## Record Schema

Every line in all JSONL files follows this schema:

```json
{
  "text": "...",
  "load_score": 55,
  "load_label": "Medium",
  "domain": "news",
  "source": "onestop",
  "article_id": "onestop_00042"
}
```

- `load_score`: integer 0–100
- `load_label`: `"Low"` (score < 40), `"Medium"` (40–70), `"High"` (> 70)
- `domain`: `"news"`, `"legal"`, `"medical"`, or `"general"`
- `source`: `"onestop"` or `"clear"`

## Datasets

### 1. OneStopEnglish (auto-downloaded)

Run:
```bash
python scripts/download_data.py
```

Downloads from https://github.com/nishkalavallabhi/OneStopEnglishCorpus.
The corpus contains the same news articles at three reading levels:

| Level | Load Score | Load Label |
|-------|-----------|------------|
| Elementary | 20 | Low |
| Intermediate | 55 | Medium |
| Advanced | 85 | High |

### 2. CLEAR Corpus (manual download required)

The CLEAR corpus contains plain and complex versions of legal and medical documents.
Access requires completing a form at:

https://github.com/lauramanor/clear-corpus

Once you have the CSV file, place it at:
```
data/raw/clear.csv
```

Then run:
```bash
python scripts/load_clear.py
```

| Complexity | Load Score | Load Label |
|------------|-----------|------------|
| plain | 15 | Low |
| complex | 80 | High |

## Creating Splits

After downloading data, run:
```bash
python scripts/prepare_dataset.py
```

This creates stratified 80/10/10 train/val/test splits in `data/processed/`.

Expected totals with both datasets: ~10,000–13,000 samples after deduplication.
