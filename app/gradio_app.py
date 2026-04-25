from __future__ import annotations

import io
import os
import tempfile

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Lazy-init pipeline so app starts quickly
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline.end_to_end import CognitiveLoadPipeline
        _pipeline = CognitiveLoadPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Tab 1 — Analyze
# ---------------------------------------------------------------------------

def analyze_text(text: str):
    """Run detection and return score, label, radar chart, HTML highlight."""
    if not text or not text.strip():
        return 0.0, {}, None, "<p>Please enter some text.</p>", {}

    pipeline = get_pipeline()
    detection = pipeline.predictor.predict(text)

    load_score = detection["load_score"]
    load_label = detection["load_label"]
    confidence = detection["confidence"]
    dimensions = detection["dimensions"]
    sentence_scores = detection["sentence_scores"]

    # Label dict for gr.Label
    label_dict = {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0,
    }
    label_dict[load_label] = confidence

    # Radar chart
    from src.utils.visualization import build_radar_chart, highlight_sentences
    radar = build_radar_chart(dimensions)

    # Sentence highlighting
    html_out = highlight_sentences(text, sentence_scores)

    # Store detection result in state
    state = {
        "text": text,
        "detection": detection,
    }

    return load_score, label_dict, radar, html_out, state


# ---------------------------------------------------------------------------
# Tab 2 — Reframe
# ---------------------------------------------------------------------------

def reframe_text(text: str, state: dict):
    """Reframe the text and show comparison."""
    if not text or not text.strip():
        return text, "", [], 0.0, state

    pipeline = get_pipeline()

    # Use detection from state if available and text matches
    if state and state.get("text") == text and "detection" in state:
        detection = state["detection"]
    else:
        detection = pipeline.predictor.predict(text)

    # Always reframe regardless of threshold (user explicitly asked)
    candidates = pipeline.rewriter.rewrite(text)
    eval_result = pipeline.evaluator.evaluate(
        original=text,
        candidates=candidates,
        reference=None,
    )

    best_text = eval_result["best_candidate"]
    final_detection = pipeline.predictor.predict(best_text)

    # Build comparison table
    from src.utils.visualization import build_score_comparison_table
    rows = build_score_comparison_table(detection, final_detection)
    df_rows = [
        [r["Metric"], r["Before"], r["After"], r["Change"]]
        for r in rows
    ]

    load_delta = eval_result["load_delta"]

    new_state = {
        "text": text,
        "detection": detection,
        "reframed_text": best_text,
        "final_detection": final_detection,
    }

    return text, best_text, df_rows, round(load_delta, 1), new_state


# ---------------------------------------------------------------------------
# Tab 3 — Batch
# ---------------------------------------------------------------------------

def run_batch(file_obj):
    """Process uploaded CSV and return results."""
    if file_obj is None:
        return [], None

    pipeline = get_pipeline()

    try:
        df = pd.read_csv(file_obj.name)
    except Exception as exc:
        return [[f"Error reading CSV: {exc}"]], None

    # Find text column
    text_col = None
    for c in ["text", "content", "sentence", "passage", "document"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None and len(df.columns) > 0:
        text_col = df.columns[0]
    if text_col is None:
        return [["No text column found"]], None

    results: list[dict] = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        if len(text.strip()) < 5:
            continue
        detection = pipeline.predictor.predict(text)
        results.append(
            {
                "text": text[:120] + "…" if len(text) > 120 else text,
                "load_score": detection["load_score"],
                "load_label": detection["load_label"],
                "confidence": detection["confidence"],
                "method": detection["method"],
            }
        )

    if not results:
        return [["No valid rows found"]], None

    result_df = pd.DataFrame(results)

    # Save to temp file for download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    result_df.to_csv(tmp.name, index=False)
    tmp.close()

    preview = result_df.values.tolist()
    headers = list(result_df.columns)
    table_data = [headers] + preview

    return table_data, tmp.name


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Cognitive Load Detector") as demo:
        gr.Markdown(
            "# 🧠 Cognitive Load Detection & Reframing\n"
            "Detect information overload in text, visualize it, and automatically rewrite high-load content."
        )

        state = gr.State({})

        with gr.Tabs():

            # ---- TAB 1: ANALYZE ----
            with gr.Tab("Analyze"):
                gr.Markdown("### Paste text to analyze its cognitive load level.")
                with gr.Row():
                    with gr.Column(scale=2):
                        input_text = gr.Textbox(
                            lines=10,
                            label="Input text",
                            placeholder="Paste any news article, legal document, or academic text here…",
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=1):
                        score_out = gr.Number(label="Load score (0–100)")
                        label_out = gr.Label(label="Load level", num_top_classes=3)

                with gr.Row():
                    radar_out = gr.Plot(label="Dimension breakdown")
                    highlight_out = gr.HTML(label="Sentence-level highlighting")

                analyze_btn.click(
                    fn=analyze_text,
                    inputs=[input_text],
                    outputs=[score_out, label_out, radar_out, highlight_out, state],
                )

            # ---- TAB 2: REFRAME ----
            with gr.Tab("Reframe"):
                gr.Markdown(
                    "### Automatically simplify high-load text using an LLM.\n"
                    "Requires Ollama running locally or a `HF_TOKEN` in your `.env`."
                )
                reframe_input = gr.Textbox(
                    lines=10,
                    label="Input text",
                    placeholder="Paste complex text here…",
                )
                reframe_btn = gr.Button("Reframe text", variant="primary")

                with gr.Row():
                    original_box = gr.Textbox(label="Original", lines=8, interactive=False)
                    reframed_box = gr.Textbox(label="Reframed", lines=8, interactive=False)

                comparison_table = gr.Dataframe(
                    headers=["Metric", "Before", "After", "Change"],
                    label="Score comparison",
                )
                delta_out = gr.Number(label="Load score reduction (negative = better)")

                reframe_btn.click(
                    fn=reframe_text,
                    inputs=[reframe_input, state],
                    outputs=[original_box, reframed_box, comparison_table, delta_out, state],
                )

            # ---- TAB 3: BATCH ----
            with gr.Tab("Batch"):
                gr.Markdown(
                    "### Upload a CSV file with a `text` column to analyze multiple documents at once."
                )
                file_input = gr.File(label="Upload CSV", file_types=[".csv"])
                batch_btn = gr.Button("Run batch", variant="primary")
                batch_table = gr.Dataframe(label="Results preview")
                download_out = gr.File(label="Download results CSV")

                batch_btn.click(
                    fn=run_batch,
                    inputs=[file_input],
                    outputs=[batch_table, download_out],
                )

            # ---- TAB 4: ABOUT ----
            with gr.Tab("About"):
                gr.Markdown(
                    """
## About this project

**Cognitive Load Detection & Reframing System** — a production-ready NLP pipeline that measures
how cognitively demanding any piece of text is, then automatically rewrites high-load passages
into simpler, equally informative prose.

### How it works
1. **Feature extraction** — syntactic (dependency depth, clause count), lexical (rare-word ratio,
   nominalizations), density (NER, concept density), coherence (sentence embedding similarity drops)
2. **Detection** — DeBERTa-v3-base dual-head model (regression + classification). Falls back to
   a calibrated heuristic when no trained model is present.
3. **Reframing** — Mistral-7B-Instruct via local Ollama or HuggingFace Inference API
4. **Evaluation** — SARI (custom implementation) + BERTScore + load-delta composite ranking

### Datasets
- **OneStopEnglish** — same news articles at Elementary / Intermediate / Advanced reading levels
- **CLEAR Corpus** — plain vs complex legal and medical document pairs (form required)

### Models
| Component | Model |
|-----------|-------|
| Syntactic + lexical + density | spaCy `en_core_web_lg` |
| Coherence | `all-mpnet-base-v2` (sentence-transformers) |
| Detection | `microsoft/deberta-v3-base` |
| Rewriting | `mistralai/Mistral-7B-Instruct-v0.2` |
| Evaluation | BERTScore + custom SARI |

### Python 3.13 compatibility
- Uses `en_core_web_lg` (not `en_core_web_trf`) — no spacy-transformers conflict
- SARI implemented natively — no `easse` dependency
- No deprecated stdlib modules (`audioop`, `distutils`, `pkg_resources`)

### GitHub
[github.com/your-org/cognitive-load-nlp](#) *(placeholder)*
                    """
                )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=False)
