from __future__ import annotations

import html

from src.utils.text_utils import split_sentences


def _score_to_color(score: float) -> str:
    if score < 40:
        return "#d4edda"
    if score <= 70:
        return "#fff3cd"
    return "#f8d7da"


def _score_to_text_color(score: float) -> str:
    if score < 40:
        return "#155724"
    if score <= 70:
        return "#856404"
    return "#721c24"


def highlight_sentences(text: str, sentence_scores: list[float]) -> str:
    """Return HTML with each sentence color-coded by cognitive load score.

    - score < 40: green tint (#d4edda)
    - score 40-70: yellow tint (#fff3cd)
    - score > 70: red tint (#f8d7da)
    Includes hover tooltip showing the exact score.
    """
    sentences = split_sentences(text)

    # Align scores to sentences (pad or truncate)
    n = len(sentences)
    if len(sentence_scores) < n:
        last = sentence_scores[-1] if sentence_scores else 50.0
        sentence_scores = list(sentence_scores) + [last] * (n - len(sentence_scores))
    else:
        sentence_scores = list(sentence_scores[:n])

    parts: list[str] = []
    for sent, score in zip(sentences, sentence_scores):
        bg = _score_to_color(score)
        fg = _score_to_text_color(score)
        label = "Low" if score < 40 else ("Medium" if score <= 70 else "High")
        escaped = html.escape(sent)
        span = (
            f'<span style="background-color:{bg};color:{fg};padding:2px 4px;'
            f'border-radius:3px;margin:1px;display:inline;" '
            f'title="Load score: {score:.1f} ({label})">'
            f"{escaped}</span>"
        )
        parts.append(span)

    legend = (
        '<div style="margin-top:10px;font-size:0.85em;">'
        '<span style="background:#d4edda;padding:2px 6px;border-radius:3px;">Low (&lt;40)</span>&nbsp;'
        '<span style="background:#fff3cd;padding:2px 6px;border-radius:3px;">Medium (40–70)</span>&nbsp;'
        '<span style="background:#f8d7da;padding:2px 6px;border-radius:3px;">High (&gt;70)</span>'
        "</div>"
    )

    return "<div style='line-height:1.8;'>" + " ".join(parts) + legend + "</div>"


def build_radar_chart(dimensions: dict[str, float]) -> dict:
    """Return a plotly figure dict for a radar chart of all dimensions."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return {}

    categories = list(dimensions.keys())
    values = [float(v) for v in dimensions.values()]

    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(66, 133, 244, 0.2)",
                line=dict(color="rgba(66, 133, 244, 0.9)", width=2),
                name="Load score",
            )
        ],
        layout=go.Layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
            ),
            showlegend=False,
            title="Cognitive Load Dimensions",
            margin=dict(t=60, b=20, l=40, r=40),
        ),
    )

    return fig


def build_score_comparison_table(before: dict, after: dict) -> list[dict]:
    """Return list of row dicts for a Gradio Dataframe component.

    Each row: {"Metric": str, "Before": float, "After": float, "Change": str}
    """
    rows: list[dict] = []

    def fmt_change(b: float, a: float) -> str:
        delta = a - b
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.1f}"

    rows.append(
        {
            "Metric": "Load Score",
            "Before": round(before.get("load_score", 0), 1),
            "After": round(after.get("load_score", 0), 1),
            "Change": fmt_change(before.get("load_score", 0), after.get("load_score", 0)),
        }
    )

    dims_before = before.get("dimensions", {})
    dims_after = after.get("dimensions", {})
    for dim in ["lexical", "syntactic", "density", "coherence"]:
        b_val = dims_before.get(dim, 0.0)
        a_val = dims_after.get(dim, 0.0)
        rows.append(
            {
                "Metric": f"  {dim.capitalize()} dim.",
                "Before": round(b_val, 1),
                "After": round(a_val, 1),
                "Change": fmt_change(b_val, a_val),
            }
        )

    rows.append(
        {
            "Metric": "Confidence",
            "Before": round(before.get("confidence", 0), 3),
            "After": round(after.get("confidence", 0), 3),
            "Change": fmt_change(
                before.get("confidence", 0), after.get("confidence", 0)
            ),
        }
    )

    return rows
