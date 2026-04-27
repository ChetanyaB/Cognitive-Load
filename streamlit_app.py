# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# import streamlit as st
# import pandas as pd
# import tempfile
# from dotenv import load_dotenv

# load_dotenv()

# from src.reframing.gpt_engine import gpt_analyze, gpt_batch_analyze, gpt_reframe

# # ─────────────────────────────────────────────
# #  Page config
# # ─────────────────────────────────────────────

# st.set_page_config(
#     page_title="Cognitive Load Detector",
#     page_icon="🧠",
#     layout="wide",
# )

# st.title("🧠 Cognitive Load Detection & Reframing")
# st.caption("Detects information overload and rewrites high-load content.")

# # ─────────────────────────────────────────────
# #  Helpers
# # ─────────────────────────────────────────────

# def score_to_color(score):
#     if score < 40:
#         return "🟢"
#     if score <= 70:
#         return "🟡"
#     return "🔴"


# def label_to_badge(label):
#     colors = {"Low": "green", "Medium": "orange", "High": "red"}
#     color = colors.get(label, "gray")
#     return f":{color}[**{label}**]"


# def render_highlighted_sentences(text, sentence_scores):
#     from src.utils.text_utils import split_sentences
#     sentences = split_sentences(text)

#     if len(sentence_scores) < len(sentences):
#         last = sentence_scores[-1] if sentence_scores else 50.0
#         sentence_scores = list(sentence_scores) + [last] * (len(sentences) - len(sentence_scores))

#     html_parts = []
#     for sent, score in zip(sentences, sentence_scores):
#         if score < 40:
#             bg = "#d4edda"
#             fg = "#155724"
#         elif score <= 70:
#             bg = "#fff3cd"
#             fg = "#856404"
#         else:
#             bg = "#f8d7da"
#             fg = "#721c24"

#         html_parts.append(
#             f'<span style="background-color:{bg};color:{fg};padding:3px 6px;'
#             f'border-radius:4px;margin:2px;display:inline;" '
#             f'title="Score: {score:.0f}">{sent}</span>'
#         )

#     legend = (
#         '<div style="margin-top:12px;font-size:0.85em;">'
#         '<span style="background:#d4edda;padding:2px 8px;border-radius:3px;">🟢 Low (&lt;40)</span>&nbsp;&nbsp;'
#         '<span style="background:#fff3cd;padding:2px 8px;border-radius:3px;">🟡 Medium (40–70)</span>&nbsp;&nbsp;'
#         '<span style="background:#f8d7da;padding:2px 8px;border-radius:3px;">🔴 High (&gt;70)</span>'
#         '</div>'
#     )

#     return "<div style='line-height:2.2;'>" + " ".join(html_parts) + legend + "</div>"


# def render_radar_chart(dimensions):
#     try:
#         import plotly.graph_objects as go
#         categories = list(dimensions.keys())
#         values = [float(v) for v in dimensions.values()]
#         categories_closed = categories + [categories[0]]
#         values_closed = values + [values[0]]
#         fig = go.Figure(
#             data=[go.Scatterpolar(
#                 r=values_closed,
#                 theta=categories_closed,
#                 fill="toself",
#                 fillcolor="rgba(66,133,244,0.2)",
#                 line=dict(color="rgba(66,133,244,0.9)", width=2),
#             )],
#             layout=go.Layout(
#                 polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
#                 showlegend=False,
#                 margin=dict(t=40, b=20, l=40, r=40),
#                 height=320,
#             ),
#         )
#         return fig
#     except Exception:
#         return None


# # ─────────────────────────────────────────────
# #  Tabs
# # ─────────────────────────────────────────────

# tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "✏️ Reframe", "📊 Batch"])


# # ══════════════════════════════════════════════
# #  TAB 1 — ANALYZE
# # ══════════════════════════════════════════════

# with tab1:
#     st.subheader("Analyze Cognitive Load")
#     st.write("Paste any text below")

#     input_text = st.text_area(
#         "Input text",
#         height=220,
#         placeholder="Paste a news article, legal clause, academic abstract, or any text here…",
#         key="analyze_input",
#     )

#     analyze_btn = st.button("Analyze", type="primary", key="analyze_btn")

#     if analyze_btn:
#         if not input_text.strip():
#             st.warning("Please enter some text before analyzing.")
#         else:
#             with st.spinner("Sending for analysis…"):
#                 try:
#                     result = gpt_analyze(input_text)

#                     load_score = result["load_score"]
#                     load_label = result["load_label"]
#                     confidence = result["confidence"]
#                     dimensions = result["dimensions"]
#                     sentence_scores = result.get("sentence_scores", [load_score])
#                     explanation = result.get("explanation", "")

#                     # ── Metric cards ──
#                     st.divider()
#                     col1, col2, col3, col4 = st.columns(4)

#                     with col1:
#                         st.metric(
#                             label="Load Score",
#                             value=f"{load_score} / 100",
#                             delta=None,
#                         )
#                     with col2:
#                         st.metric(label="Load Level", value=f"{score_to_color(load_score)} {load_label}")
#                     with col3:
#                         st.metric(label="Confidence", value=f"{confidence:.0%}")
#                     with col4:
#                         st.metric(label="Method", value="GPT")

#                     # ── Progress bar ──
#                     st.markdown("**Overall cognitive load:**")
#                     bar_color = "normal" if load_score < 40 else ("off" if load_score > 70 else "normal")
#                     st.progress(int(load_score) / 100)

#                     # ── Explanation ──
#                     if explanation:
#                         st.info(f"💡 **Explanation** {explanation}")

#                     # ── Dimension breakdown + radar ──
#                     st.divider()
#                     col_left, col_right = st.columns([1, 1])

#                     with col_left:
#                         st.markdown("**Dimension Scores**")
#                         dim_data = {
#                             "Dimension": list(dimensions.keys()),
#                             "Score": [int(v) for v in dimensions.values()],
#                         }
#                         dim_df = pd.DataFrame(dim_data)
#                         st.dataframe(
#                             dim_df,
#                             hide_index=True,
#                             use_container_width=True,
#                             column_config={
#                                 "Score": st.column_config.ProgressColumn(
#                                     "Score",
#                                     min_value=0,
#                                     max_value=100,
#                                     format="%d",
#                                 )
#                             },
#                         )

#                     with col_right:
#                         fig = render_radar_chart(dimensions)
#                         if fig:
#                             st.plotly_chart(fig, use_container_width=True)

#                     # ── Sentence highlighting ──
#                     st.divider()
#                     st.markdown("**Sentence-level highlighting**")
#                     try:
#                         html = render_highlighted_sentences(input_text, sentence_scores)
#                         st.html(html)
#                     except Exception:
#                         st.write(input_text)

#                 except Exception as exc:
#                     st.error(f"hidden error: {exc}")


# # ══════════════════════════════════════════════
# #  TAB 2 — REFRAME
# # ══════════════════════════════════════════════

# with tab2:
#     st.subheader("Reframe High-Load Text")
#     st.write("Model rewrites the text to reduce cognitive load, then re-analyzes and scores both versions.")

#     reframe_input = st.text_area(
#         "Input text",
#         height=220,
#         placeholder="Paste complex text here — legal clauses, academic abstracts, technical documentation…",
#         key="reframe_input",
#     )

#     reframe_btn = st.button("Reframe Text", type="primary", key="reframe_btn")

#     if reframe_btn:
#         if not reframe_input.strip():
#             st.warning("Please enter some text before reframing.")
#         else:
#             with st.spinner("Model is rewriting and evaluating…"):
#                 try:
#                     result = gpt_reframe(reframe_input)

#                     original_text = result["original_text"]
#                     reframed_text = result["reframed_text"]
#                     orig_analysis = result["original_analysis"]
#                     ref_analysis = result["reframed_analysis"]
#                     reframe_scores = result["reframe_scores"]

#                     orig_score = orig_analysis["load_score"]
#                     ref_score = ref_analysis["load_score"]
#                     load_delta = ref_score - orig_score

#                     # ── Summary banner ──
#                     st.divider()
#                     if load_delta < 0:
#                         st.success(
#                             f"✅ Load reduced by **{abs(load_delta):.0f} points** — "
#                             f"{orig_analysis['load_label']} → {ref_analysis['load_label']}"
#                         )
#                     elif load_delta == 0:
#                         st.info("Load score unchanged after reframing.")
#                     else:
#                         st.warning(
#                             f"⚠️ Load increased by {load_delta:.0f} points. Try again."
#                         )

#                     # ── Side-by-side text ──
#                     col_orig, col_ref = st.columns(2)
#                     with col_orig:
#                         st.markdown(
#                             f"**Original** &nbsp; {score_to_color(orig_score)} Score: `{orig_score}`"
#                         )
#                         st.text_area(
#                             "original_text_display",
#                             value=original_text,
#                             height=280,
#                             disabled=True,
#                             label_visibility="collapsed",
#                             key="orig_display",
#                         )
#                     with col_ref:
#                         st.markdown(
#                             f"**Reframed** &nbsp; {score_to_color(ref_score)} Score: `{ref_score}`"
#                         )
#                         st.text_area(
#                             "reframed_text_display",
#                             value=reframed_text,
#                             height=280,
#                             disabled=True,
#                             label_visibility="collapsed",
#                             key="ref_display",
#                         )

#                     # ── Score comparison table ──
#                     st.divider()
#                     st.markdown("**Score Comparison**")

#                     orig_dims = orig_analysis.get("dimensions", {})
#                     ref_dims = ref_analysis.get("dimensions", {})

#                     def delta_str(b, a):
#                         d = a - b
#                         sign = "+" if d > 0 else ""
#                         return f"{sign}{d:.0f}"

#                     comparison_rows = [
#                         {"Metric": "Load Score", "Before": orig_score, "After": ref_score,
#                          "Change": delta_str(orig_score, ref_score)},
#                         {"Metric": "Syntactic", "Before": orig_dims.get("syntactic", 0),
#                          "After": ref_dims.get("syntactic", 0),
#                          "Change": delta_str(orig_dims.get("syntactic", 0), ref_dims.get("syntactic", 0))},
#                         {"Metric": "Lexical", "Before": orig_dims.get("lexical", 0),
#                          "After": ref_dims.get("lexical", 0),
#                          "Change": delta_str(orig_dims.get("lexical", 0), ref_dims.get("lexical", 0))},
#                         {"Metric": "Density", "Before": orig_dims.get("density", 0),
#                          "After": ref_dims.get("density", 0),
#                          "Change": delta_str(orig_dims.get("density", 0), ref_dims.get("density", 0))},
#                         {"Metric": "Coherence", "Before": orig_dims.get("coherence", 0),
#                          "After": ref_dims.get("coherence", 0),
#                          "Change": delta_str(orig_dims.get("coherence", 0), ref_dims.get("coherence", 0))},
#                         {"Metric": "SARI", "Before": "—",
#                          "After": f"{reframe_scores.get('sari', 0) * 100:.1f}", "Change": ""},
#                         {"Metric": "BERTScore", "Before": "—",
#                          "After": f"{reframe_scores.get('bert_score', 0) * 100:.1f}", "Change": ""},
#                     ]

#                     st.dataframe(
#                         pd.DataFrame(comparison_rows),
#                         hide_index=True,
#                         use_container_width=True,
#                     )

#                     # ── Radar charts side by side ──
#                     st.divider()
#                     st.markdown("**Dimension Comparison**")
#                     r_col1, r_col2 = st.columns(2)
#                     with r_col1:
#                         st.caption("Before")
#                         fig_orig = render_radar_chart(orig_dims)
#                         if fig_orig:
#                             st.plotly_chart(fig_orig, use_container_width=True)
#                     with r_col2:
#                         st.caption("After")
#                         fig_ref = render_radar_chart(ref_dims)
#                         if fig_ref:
#                             st.plotly_chart(fig_ref, use_container_width=True)

#                 except Exception as exc:
#                     st.error(f"hidden error: {exc}")


# # ══════════════════════════════════════════════
# #  TAB 3 — BATCH
# # ══════════════════════════════════════════════

# with tab3:
#     st.subheader("Batch Analysis")
#     st.write(
#         "Upload a CSV file with a **`text`** column. "
#         "Each row will be analyzed individually."
#     )

#     st.info(
#         "💡 Your CSV must have a column named `text` (or `content`, `sentence`, `passage`, `document`). "
#         "Each row = one document to analyze."
#     )

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             st.write(f"**Preview** — {len(df)} rows, {len(df.columns)} columns")
#             st.dataframe(df.head(5), use_container_width=True)

#             # Find text column
#             text_col = None
#             for c in ["text", "content", "sentence", "passage", "document"]:
#                 if c in df.columns:
#                     text_col = c
#                     break
#             if text_col is None:
#                 text_col = df.columns[0]

#             st.write(f"Using column: **`{text_col}`**")

#             batch_btn = st.button("Run Batch Analysis", type="primary", key="batch_btn")

#             if batch_btn:
#                 texts = [str(r) for r in df[text_col] if str(r).strip()]
#                 st.write(f"Analyzing **{len(texts)}** texts…")

#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
#                 results_placeholder = st.empty()

#                 all_results = []
#                 for i, text in enumerate(texts):
#                     status_text.text(f"Processing {i + 1} of {len(texts)}…")
#                     try:
#                         r = gpt_analyze(text)
#                         all_results.append({
#                             "Text": text[:100] + "…" if len(text) > 100 else text,
#                             "Score": r["load_score"],
#                             "Label": r["load_label"],
#                             "Confidence": f"{r['confidence']:.0%}",
#                             "Syntactic": r["dimensions"].get("syntactic", 0),
#                             "Lexical": r["dimensions"].get("lexical", 0),
#                             "Density": r["dimensions"].get("density", 0),
#                             "Coherence": r["dimensions"].get("coherence", 0),
#                         })
#                     except Exception as exc:
#                         all_results.append({
#                             "Text": text[:100],
#                             "Score": 0,
#                             "Label": "Error",
#                             "Confidence": "—",
#                             "Syntactic": 0,
#                             "Lexical": 0,
#                             "Density": 0,
#                             "Coherence": str(exc)[:60],
#                         })
#                     progress_bar.progress((i + 1) / len(texts))

#                 status_text.text("Done!")

#                 result_df = pd.DataFrame(all_results)

#                 st.divider()
#                 st.markdown("### Results")

#                 # Summary stats
#                 s_col1, s_col2, s_col3, s_col4 = st.columns(4)
#                 with s_col1:
#                     st.metric("Total analyzed", len(result_df))
#                 with s_col2:
#                     low_count = (result_df["Label"] == "Low").sum()
#                     st.metric("🟢 Low", low_count)
#                 with s_col3:
#                     med_count = (result_df["Label"] == "Medium").sum()
#                     st.metric("🟡 Medium", med_count)
#                 with s_col4:
#                     high_count = (result_df["Label"] == "High").sum()
#                     st.metric("🔴 High", high_count)

#                 # Full results table
#                 st.dataframe(
#                     result_df,
#                     hide_index=True,
#                     use_container_width=True,
#                     column_config={
#                         "Score": st.column_config.ProgressColumn(
#                             "Score", min_value=0, max_value=100, format="%d"
#                         ),
#                     },
#                 )

#                 # Download button
#                 csv_data = result_df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     label="⬇️ Download results as CSV",
#                     data=csv_data,
#                     file_name="cognitive_load_results.csv",
#                     mime="text/csv",
#                 )

#         except Exception as exc:
#             st.error(f"Error reading CSV: {exc}")





# import streamlit as st
# import pandas as pd
# from dotenv import load_dotenv

# load_dotenv()

# from src.reframing.gpt_engine import gpt_analyze, gpt_batch_analyze, gpt_reframe

# # ─────────────────────────────────────────────
# # Page Config
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="Cognitive Load Analyzer",
#     page_icon="🧠",
#     layout="wide",
# )

# # ─────────────────────────────────────────────
# # Custom Styling
# # ─────────────────────────────────────────────
# st.markdown("""
# <style>
# .main {
#     background-color: #0f172a;
# }
# .block-container {
#     padding-top: 2rem;
# }

# .card {
#     padding: 1.5rem;
#     border-radius: 12px;
#     background: #111827;
#     border: 1px solid #1f2937;
# }

# .metric-card {
#     padding: 1rem;
#     border-radius: 10px;
#     background: #1f2937;
#     text-align: center;
# }

# .title {
#     font-size: 2.2rem;
#     font-weight: 700;
# }

# .subtitle {
#     color: #9ca3af;
#     margin-bottom: 2rem;
# }

# textarea {
#     border-radius: 10px !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────────────────────────
# # Sidebar
# # ─────────────────────────────────────────────
# st.sidebar.title("🧠 Cognitive Toolkit")
# mode = st.sidebar.radio(
#     "Choose Mode",
#     ["Analyze", "Reframe", "Batch"]
# )

# st.sidebar.markdown("---")
# st.sidebar.caption("Designed for understanding complex text faster")

# # ─────────────────────────────────────────────
# # Header
# # ─────────────────────────────────────────────
# st.markdown('<div class="title">Cognitive Load Analyzer</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">Understand and simplify complex information instantly</div>', unsafe_allow_html=True)

# # ─────────────────────────────────────────────
# # Helper
# # ─────────────────────────────────────────────
# def score_color(score):
#     if score < 40:
#         return "🟢 Low"
#     elif score <= 70:
#         return "🟡 Medium"
#     return "🔴 High"

# # ══════════════════════════════════════════════
# # ANALYZE
# # ══════════════════════════════════════════════
# if mode == "Analyze":
#     st.markdown("### 🔍 Analyze Text")

#     text = st.text_area(
#         "Paste your text",
#         height=200,
#         placeholder="Paste any article, document, or paragraph..."
#     )

#     if st.button("Analyze", use_container_width=True):
#         if not text.strip():
#             st.warning("Please enter text")
#         else:
#             with st.spinner("Analyzing..."):
#                 result = gpt_analyze(text)

#             score = result["load_score"]
#             label = result["load_label"]
#             confidence = result["confidence"]

#             st.markdown("### Results")

#             col1, col2, col3 = st.columns(3)

#             with col1:
#                 st.markdown(f"""
#                 <div class="metric-card">
#                 <h3>{score}/100</h3>
#                 <p>Load Score</p>
#                 </div>
#                 """, unsafe_allow_html=True)

#             with col2:
#                 st.markdown(f"""
#                 <div class="metric-card">
#                 <h3>{score_color(score)}</h3>
#                 <p>Complexity</p>
#                 </div>
#                 """, unsafe_allow_html=True)

#             with col3:
#                 st.markdown(f"""
#                 <div class="metric-card">
#                 <h3>{confidence:.0%}</h3>
#                 <p>Confidence</p>
#                 </div>
#                 """, unsafe_allow_html=True)

#             st.progress(score / 100)

#             if "explanation" in result:
#                 st.info(result["explanation"])

# # ══════════════════════════════════════════════
# # REFRAME
# # ══════════════════════════════════════════════
# elif mode == "Reframe":
#     st.markdown("### ✏️ Simplify Text")

#     text = st.text_area(
#         "Enter complex text",
#         height=200,
#         placeholder="Paste difficult content here..."
#     )

#     if st.button("Simplify", use_container_width=True):
#         if not text.strip():
#             st.warning("Please enter text")
#         else:
#             with st.spinner("Rewriting..."):
#                 result = gpt_reframe(text)

#             orig = result["original_analysis"]["load_score"]
#             ref = result["reframed_analysis"]["load_score"]

#             delta = orig - ref

#             if delta > 0:
#                 st.success(f"Reduced complexity by {delta} points")
#             else:
#                 st.warning("No improvement detected")

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.markdown("#### Original")
#                 st.text_area("", result["original_text"], height=250)

#             with col2:
#                 st.markdown("#### Simplified")
#                 st.text_area("", result["reframed_text"], height=250)

# # ══════════════════════════════════════════════
# # BATCH
# # ══════════════════════════════════════════════
# elif mode == "Batch":
#     st.markdown("### 📊 Batch Analysis")

#     file = st.file_uploader("Upload CSV", type=["csv"])

#     if file:
#         df = pd.read_csv(file)

#         st.dataframe(df.head(), use_container_width=True)

#         text_col = df.columns[0]

#         if st.button("Run Analysis", use_container_width=True):
#             results = []

#             progress = st.progress(0)

#             for i, row in enumerate(df[text_col]):
#                 r = gpt_analyze(str(row))

#                 results.append({
#                     "Text": str(row)[:80],
#                     "Score": r["load_score"],
#                     "Label": r["load_label"]
#                 })

#                 progress.progress((i + 1) / len(df))

#             result_df = pd.DataFrame(results)

#             st.success("Completed")

#             st.dataframe(result_df, use_container_width=True)

#             st.download_button(
#                 "Download CSV",
#                 result_df.to_csv(index=False),
#                 "results.csv"
#             )

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.reframing.gpt_engine import gpt_analyze, gpt_batch_analyze, gpt_reframe

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive Load Analyzer",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background-color: #111827;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #1f2937;
}
.title {
    font-size: 2.2rem;
    font-weight: 700;
}
.subtitle {
    color: #9ca3af;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.title("🧠 Cognitive Toolkit")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Analyze", "✏️ Simplify", "📊 Batch"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Understand and simplify complex information")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="title">🧠 Cognitive Load Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze, understand, and simplify complex text</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS (UNCHANGED)
# ─────────────────────────────────────────────
def score_to_color(score):
    if score < 40:
        return "🟢"
    if score <= 70:
        return "🟡"
    return "🔴"


def render_highlighted_sentences(text, sentence_scores):
    from src.utils.text_utils import split_sentences
    sentences = split_sentences(text)

    if len(sentence_scores) < len(sentences):
        last = sentence_scores[-1] if sentence_scores else 50.0
        sentence_scores = list(sentence_scores) + [last] * (len(sentences) - len(sentence_scores))

    html_parts = []
    for sent, score in zip(sentences, sentence_scores):
        if score < 40:
            bg = "#d4edda"
        elif score <= 70:
            bg = "#fff3cd"
        else:
            bg = "#f8d7da"

        html_parts.append(
            f'<span style="background-color:{bg};padding:4px 8px;'
            f'border-radius:6px;margin:3px;display:inline;" '
            f'title="Score: {score:.0f}">{sent}</span>'
        )

    return "<div style='line-height:2.2;'>" + " ".join(html_parts) + "</div>"


def render_radar_chart(dimensions):
    try:
        import plotly.graph_objects as go
        categories = list(dimensions.keys())
        values = [float(v) for v in dimensions.values()]
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig = go.Figure(
            data=[go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself"
            )]
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=320
        )

        return fig
    except:
        return None


# ══════════════════════════════════════════════
# ANALYZE PAGE
# ══════════════════════════════════════════════
if page == "🔍 Analyze":
    st.subheader("Analyze Text Complexity")

    text = st.text_area("Enter text", height=220)

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter text")
        else:
            with st.spinner("Analyzing..."):
                result = gpt_analyze(text)

            score = result["load_score"]
            label = result["load_label"]
            confidence = result["confidence"]
            dimensions = result["dimensions"]

            col1, col2, col3 = st.columns(3)

            col1.markdown(f"<div class='metric-card'><h3>{score}/100</h3><p>Score</p></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-card'><h3>{score_to_color(score)} {label}</h3><p>Level</p></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-card'><h3>{confidence:.0%}</h3><p>Confidence</p></div>", unsafe_allow_html=True)

            st.progress(score / 100)

            colL, colR = st.columns(2)

            with colL:
                st.dataframe(pd.DataFrame({
                    "Dimension": list(dimensions.keys()),
                    "Score": list(dimensions.values())
                }), use_container_width=True)

            with colR:
                fig = render_radar_chart(dimensions)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Sentence Analysis")
            st.html(render_highlighted_sentences(text, result.get("sentence_scores", [score])))


# ══════════════════════════════════════════════
# SIMPLIFY PAGE
# ══════════════════════════════════════════════
elif page == "✏️ Simplify":
    st.subheader("Simplify Complex Text")

    text = st.text_area("Enter text", height=220)

    if st.button("Simplify", type="primary"):
        if not text.strip():
            st.warning("Enter text first")
        else:
            with st.spinner("Processing..."):
                result = gpt_reframe(text)

            orig = result["original_analysis"]["load_score"]
            ref = result["reframed_analysis"]["load_score"]

            delta = ref - orig

            if delta < 0:
                st.success(f"Reduced complexity by {abs(delta):.0f} points")
            else:
                st.warning("No improvement detected")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Original ({orig})**")
                st.text_area("", result["original_text"], height=260)

            with col2:
                st.markdown(f"**Simplified ({ref})**")
                st.text_area("", result["reframed_text"], height=260)


# ══════════════════════════════════════════════
# BATCH PAGE
# ══════════════════════════════════════════════
elif page == "📊 Batch":
    st.subheader("Batch Processing")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head(), use_container_width=True)

        text_col = df.columns[0]

        if st.button("Run Analysis", type="primary"):
            results = []
            progress = st.progress(0)

            for i, row in enumerate(df[text_col]):
                r = gpt_analyze(str(row))

                results.append({
                    "Text": str(row)[:80],
                    "Score": r["load_score"],
                    "Label": r["load_label"]
                })

                progress.progress((i + 1) / len(df))

            result_df = pd.DataFrame(results)

            st.success("Completed")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Download CSV",
                result_df.to_csv(index=False),
                "results.csv"
            )