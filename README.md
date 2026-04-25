# Cognitive Load Detection & Reframing System

A production-ready NLP system that detects cognitive load in digital text and automatically rewrites high-load content using LLMs.

## What It Does

1. Takes any input text (news, legal, medical, academic)
2. Detects cognitive load across multiple dimensions (lexical, syntactic, semantic, information density)
3. Outputs a load score (0–100) and a label (Low / Medium / High) with per-dimension breakdown
4. If load is HIGH, automatically reframes the text using an LLM to a simpler version
5. Evaluates the reframe using SARI + BERTScore and confirms load dropped
6. Exposes everything via a Gradio demo with span-level highlighting and a radar chart

---

## Setup

```bash
# 1. Unzip and enter the project
cd cognitive_load_nlp

# 2. Create virtual environment (Python 3.13)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Download OneStopEnglish (automatic)
python scripts/download_data.py

# 5. (Optional) Add CLEAR corpus
# Fill form at: https://github.com/lauramanor/clear-corpus
# Place file at: data/raw/clear.csv
# Then run:
python scripts/load_clear.py

# 6. Prepare unified dataset
python scripts/prepare_dataset.py

# 7. Launch demo (works immediately without fine-tuning)
python app/gradio_app.py
```

## Fine-tuning (optional)

```bash
python -m src.detection.trainer --epochs 5 --batch_size 16 --output_dir models/detector/
```

## Run Tests

```bash
pip install pytest
pytest tests/
```

---

## Notes for Python 3.13 Users

- This project is built and tested on Python 3.13
- Uses `en_core_web_lg` (not `en_core_web_trf`) to avoid spacy-transformers conflicts
- SARI is implemented natively — no `easse` dependency
- All deprecated stdlib modules (audioop, distutils) are avoided

---

## Architecture

```
Input Text
    │
    ▼
Feature Extraction (syntactic + lexical + density + coherence)
    │
    ▼
Cognitive Load Predictor (DeBERTa-v3 + feature fusion, or heuristic fallback)
    │
    ▼
Load Score & Label (0–100, Low/Medium/High)
    │
    ├─ If LOW/MEDIUM → Return result
    │
    └─ If HIGH → TextRewriter (Mistral via HF API or Ollama)
                    │
                    ▼
                Candidates × 3 temperatures
                    │
                    ▼
                RewriteEvaluator (SARI + BERTScore + load delta)
                    │
                    ▼
                Best rewrite + evaluation scores
```

## Datasets

### OneStopEnglish
- Source: https://github.com/nishkalavallabhi/OneStopEnglishCorpus
- Auto-downloaded by `scripts/download_data.py`
- 3 reading levels: Elementary (load 20), Intermediate (55), Advanced (85)

### CLEAR Corpus
- Source: https://github.com/lauramanor/clear-corpus
- Requires filling a form to access
- Place downloaded CSV at `data/raw/clear.csv`
- plain documents → load 15, complex → load 80

## Models Used

- **Detection**: microsoft/deberta-v3-base (with feature fusion)
- **Coherence**: all-mpnet-base-v2 (sentence-transformers)
- **Rewriting**: mistralai/Mistral-7B-Instruct-v0.2 (via HF API or local Ollama)
- **Evaluation**: BERTScore, SARI (custom implementation)

## Environment Variables

Copy `.env.example` to `.env` and fill in your tokens:

```bash
cp .env.example .env
```

- `HF_TOKEN`: HuggingFace token for Mistral API access
- `MODEL_NAME`: HuggingFace model ID for rewriting
- `OLLAMA_URL`: Local Ollama endpoint (auto-detected)
