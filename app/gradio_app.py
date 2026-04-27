from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.reframing.gpt_engine import gpt_analyze, gpt_batch_analyze, gpt_reframe


# ─────────────────────────────────────────────
#  Tab 1 — Analyze
# ─────────────────────────────────────────────

def analyze_text(text):
    if not text or not text.strip():
        return 0.0, {}, None, "<p>Please enter some text.</p>", ""

    result = gpt_analyze(text)

    load_score = float(result["load_score"])
    load_label = result["load_label"]
    confidence = float(result["confidence"])
    dimensions = result["dimensions"]
    sentence_scores = result.get("sentence_scores", [load_score])
    explanation = result.get("explanation", "")

    label_dict = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    label_dict[load_label] = confidence

    try:
        from src.utils.visualization import build_radar_chart
        radar = build_radar_chart(dimensions)
    except Exception:
        radar = None

    try:
        from src.utils.text_utils import split_sentences
        from src.utils.visualization import highlight_sentences
        sentences = split_sentences(text)
        if len(sentence_scores) < len(sentences):
            sentence_scores = sentence_scores + [load_score] * (len(sentences) - len(sentence_scores))
        html_out = highlight_sentences(text, sentence_scores)
    except Exception:
        html_out = f"<p>{text}</p>"

    info_html = (
        f"<div style='padding:10px;background:#f8f9fa;border-radius:6px;margin-top:8px;'>"
        f"<b>Analysis:</b> {explanation}<br>"
        f"<b>Confidence:</b> {confidence:.0%}"
        f"</div>"
    )

    return load_score, label_dict, radar, html_out, info_html


# ─────────────────────────────────────────────
#  Tab 2 — Reframe
# ─────────────────────────────────────────────

def reframe_text(text):
    if not text or not text.strip():
        return text, "", [], 0.0, ""

    result = gpt_reframe(text)

    original_text = result["original_text"]
    reframed_text = result["reframed_text"]
    original_analysis = result["original_analysis"]
    reframed_analysis = result["reframed_analysis"]
    reframe_scores = result["reframe_scores"]

    orig_dims = original_analysis.get("dimensions", {})
    ref_dims = reframed_analysis.get("dimensions", {})

    def fmt(b, a):
        delta = a - b
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.1f}"

    table_rows = [
        ["Load Score",
         original_analysis["load_score"],
         reframed_analysis["load_score"],
         fmt(original_analysis["load_score"], reframed_analysis["load_score"])],
        ["Syntactic",
         orig_dims.get("syntactic", 0),
         ref_dims.get("syntactic", 0),
         fmt(orig_dims.get("syntactic", 0), ref_dims.get("syntactic", 0))],
        ["Lexical",
         orig_dims.get("lexical", 0),
         ref_dims.get("lexical", 0),
         fmt(orig_dims.get("lexical", 0), ref_dims.get("lexical", 0))],
        ["Density",
         orig_dims.get("density", 0),
         ref_dims.get("density", 0),
         fmt(orig_dims.get("density", 0), ref_dims.get("density", 0))],
        ["Coherence",
         orig_dims.get("coherence", 0),
         ref_dims.get("coherence", 0),
         fmt(orig_dims.get("coherence", 0), ref_dims.get("coherence", 0))],
        ["SARI", "—", round(reframe_scores.get("sari", 0) * 100, 1), ""],
        ["BERTScore", "—", round(reframe_scores.get("bert_score", 0) * 100, 1), ""],
    ]

    load_delta = float(reframe_scores.get("load_delta", 0))
    delta_color = "#28a745" if load_delta < 0 else "#dc3545"
    delta_html = (
        f"<div style='padding:10px;background:#f8f9fa;border-radius:6px;margin-top:8px;'>"
        f"<b>Load reduction:</b> "
        f"<span style='color:{delta_color};font-weight:bold;font-size:1.2em;'>"
        f"{load_delta:+.1f} points</span>"
        f" &nbsp;|&nbsp; "
        f"Original: <b>{original_analysis['load_label']}</b> → "
        f"Reframed: <b>{reframed_analysis['load_label']}</b>"
        f"</div>"
    )

    return original_text, reframed_text, table_rows, load_delta, delta_html


# ─────────────────────────────────────────────
#  Tab 3 — Batch
# ─────────────────────────────────────────────

def run_batch(file_obj):
    if file_obj is None:
        return [], None

    try:
        df = pd.read_csv(file_obj.name)
    except Exception as exc:
        return [[f"Error reading CSV: {exc}"]], None

    text_col = None
    for c in ["text", "content", "sentence", "passage", "document"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None and len(df.columns) > 0:
        text_col = df.columns[0]
    if text_col is None:
        return [["No text column found"]], None

    texts = [str(row) for row in df[text_col] if str(row).strip()]
    results = gpt_batch_analyze(texts)

    rows = []
    for r in results:
        rows.append([
            r.get("text", ""),
            r.get("load_score", 0),
            r.get("load_label", ""),
            f"{r.get('confidence', 0):.0%}",
            r.get("error", ""),
        ])

    result_df = pd.DataFrame(rows, columns=["Text", "Score", "Label", "Confidence", "Error"])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    result_df.to_csv(tmp.name, index=False)
    tmp.close()

    return rows, tmp.name


# ─────────────────────────────────────────────
#  Build App
# ─────────────────────────────────────────────

def build_app():
    # Build each tab component explicitly outside the context manager
    # to avoid Gradio version scoping issues

    analyze_tab = gr.Tab("Analyze")
    reframe_tab = gr.Tab("Reframe")
    batch_tab = gr.Tab("Batch")
    about_tab = gr.Tab("About")

    demo = gr.Blocks(title="Cognitive Load Detector")

    with demo:
        gr.Markdown(
            "# 🧠 Cognitive Load Detection & Reframing\n"
            "Detects information overload and rewrites high-load content automatically."
        )

        with gr.Tabs():

            with gr.Tab("Analyze"):
                gr.Markdown("### Paste text to analyze its cognitive load level.")
                with gr.Row():
                    with gr.Column(scale=2):
                        input_text = gr.Textbox(
                            lines=10,
                            label="Input text",
                            placeholder="Paste any text here…",
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=1):
                        score_out = gr.Number(label="Load score (0–100)")
                        label_out = gr.Label(label="Load level", num_top_classes=3)

                info_out = gr.HTML(label="Explanation")

                with gr.Row():
                    radar_out = gr.Plot(label="Dimension breakdown")
                    highlight_out = gr.HTML(label="Sentence-level highlighting")

                analyze_btn.click(
                    fn=analyze_text,
                    inputs=[input_text],
                    outputs=[score_out, label_out, radar_out, highlight_out, info_out],
                )

            with gr.Tab("Reframe"):
                gr.Markdown("### Automatically simplify high-load text.")
                reframe_input = gr.Textbox(
                    lines=10,
                    label="Input text",
                    placeholder="Paste complex text here…",
                )
                reframe_btn = gr.Button("Reframe text", variant="primary")

                with gr.Row():
                    original_box = gr.Textbox(label="Original", lines=8, interactive=False)
                    reframed_box = gr.Textbox(label="Reframed", lines=8, interactive=False)

                delta_html_out = gr.HTML(label="Load reduction summary")
                comparison_table = gr.Dataframe(
                    headers=["Metric", "Before", "After", "Change"],
                    label="Score comparison",
                )
                delta_out = gr.Number(label="Load score change (negative = better)")

                reframe_btn.click(
                    fn=reframe_text,
                    inputs=[reframe_input],
                    outputs=[original_box, reframed_box, comparison_table, delta_out, delta_html_out],
                )

            with gr.Tab("Batch"):
                gr.Markdown("### Upload a CSV with a `text` column to analyze multiple documents.")
                file_input = gr.File(label="Upload CSV", file_types=[".csv"])
                batch_btn = gr.Button("Run batch analysis", variant="primary")
                batch_table = gr.Dataframe(
                    headers=["Text", "Score", "Label", "Confidence", "Error"],
                    label="Results",
                )
                download_out = gr.File(label="Download results CSV")

                batch_btn.click(
                    fn=run_batch,
                    inputs=[file_input],
                    outputs=[batch_table, download_out],
                )

            