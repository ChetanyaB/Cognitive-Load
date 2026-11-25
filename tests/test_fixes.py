"""Regression tests for issues found during the project review (run: pytest -v)."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_texts import DEMO_TEXTS  # noqa: E402
from src.detection.predictor import CognitiveLoadPredictor  # noqa: E402

_predictor = CognitiveLoadPredictor()


def test_tc_sentence_scores_are_real_and_fast():
    text = DEMO_TEXTS["low_2"]["text"] + " " + DEMO_TEXTS["high_1"]["text"]
    t0 = time.time()
    out = _predictor.predict(text)
    assert time.time() - t0 < 30                      # was > 60 s due to runaway recursion
    assert len(set(out["sentence_scores"])) > 1       # was always [50, 50, ...]
    assert out["sentence_scores"][-1] > out["sentence_scores"][0]   # legal sentence > simple one


def test_tc_demo_texts_rank_in_order():
    s = {k: _predictor.predict(v["text"])["load_score"] for k, v in DEMO_TEXTS.items()}
    assert s["low_1"] < s["medium_1"] < s["high_1"]


def test_tc_dataset_contains_plain_text_only():
    for split in ["train", "val", "test"]:
        rows = [json.loads(l) for l in open(ROOT / "data" / "processed" / f"{split}.jsonl")]
        assert not any(r["text"].lstrip().startswith("(ROOT") for r in rows)   # 30% were parse trees


def test_tc_dataset_levels_balanced():
    rows = [json.loads(l) for l in open(ROOT / "data" / "raw" / "onestop_parsed.jsonl")]
    counts = {lab: sum(r["load_label"] == lab for r in rows) for lab in ["Low", "Medium", "High"]}
    assert counts == {"Low": 189, "Medium": 189, "High": 189}   # 189 articles x 3 levels


def test_tc_level_detection_uses_suffix():
    from download_data import detect_level
    assert detect_level("Printing-adv.txt") == "adv"   # contains "int"
    assert detect_level("Telecom-adv.txt") == "adv"    # contains "ele"
    assert detect_level("Amazon-ele.txt") == "ele"


def test_tc_gradio_app_builds():
    from app.gradio_app import build_app
    assert build_app() is not None                     # returned None before


def test_tc_analyze_works_without_api_key(monkeypatch):
    from src.reframing import gpt_engine
    monkeypatch.setattr(gpt_engine, "OPENAI_API_KEY", "")
    out = gpt_engine.gpt_analyze(DEMO_TEXTS["low_1"]["text"])
    assert out["method"].startswith("local") and out["load_label"] in {"Low", "Medium", "High"}
