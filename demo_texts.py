"""
Five example texts for immediate demo use.
Run: python demo_texts.py to print all examples with their expected load levels.
"""
from __future__ import annotations

DEMO_TEXTS: dict[str, dict] = {
    "low_1": {
        "text": "The cat sat on the mat. It was a sunny day. The cat was happy.",
        "expected_label": "Low",
        "domain": "general",
    },
    "low_2": {
        "text": (
            "Scientists say the Earth is getting warmer. This is called climate change. "
            "Burning oil, gas, and coal releases gases into the air. These gases trap heat "
            "from the sun. Governments around the world are working together to cut emissions "
            "and slow the warming down."
        ),
        "expected_label": "Low",
        "domain": "news",
    },
    "medium_1": {
        "text": (
            "Researchers at MIT have developed a new type of battery that can store twice as "
            "much energy as conventional lithium-ion cells while charging in under 15 minutes. "
            "The breakthrough relies on a solid electrolyte made from a sulfide compound, which "
            "eliminates the flammable liquid found in today's batteries. If the technology reaches "
            "commercial production, it could significantly extend the range of electric vehicles "
            "and reduce the cost of grid-scale energy storage within the next decade."
        ),
        "expected_label": "Medium",
        "domain": "news",
    },
    "high_1": {
        "text": (
            "Notwithstanding any provision herein to the contrary, the indemnifying party shall "
            "defend, indemnify, and hold harmless the indemnified party and its affiliates, "
            "officers, directors, employees, agents, successors, and assigns from and against "
            "any and all claims, damages, losses, liabilities, costs, and expenses (including "
            "reasonable attorneys' fees) arising out of or related to any third-party claim "
            "alleging that the indemnifying party's products or services, or the indemnified "
            "party's use thereof in accordance with the terms of this Agreement, infringe, "
            "misappropriate, or otherwise violate any intellectual property right or other "
            "proprietary right of such third party, provided that the indemnified party promptly "
            "notifies the indemnifying party in writing of any such claim, grants the indemnifying "
            "party sole control over the defense and settlement thereof, and provides reasonable "
            "cooperation as requested by the indemnifying party at the indemnifying party's expense."
        ),
        "expected_label": "High",
        "domain": "legal",
    },
    "high_2": {
        "text": (
            "We present a novel contrastive pre-training objective for large-scale language model "
            "alignment that leverages reward-model-free preference optimization via direct "
            "preference optimization (DPO), augmented with a distributional regularization term "
            "derived from the Kullback–Leibler divergence between the policy and the reference "
            "distribution. Our method empirically reduces reward hacking while maintaining "
            "generation diversity, as measured by self-BLEU and distinct-n metrics across "
            "benchmark summarization, dialogue, and instruction-following tasks. We further "
            "demonstrate that iterative preference data collection with an adaptive sampling "
            "temperature scheduler yields monotonically improving win-rates against both "
            "supervised fine-tuning baselines and prior RLHF-trained checkpoints on the "
            "Anthropic HH and OpenAI WebGPT comparison datasets, achieving state-of-the-art "
            "alignment performance with a 30% reduction in annotation cost."
        ),
        "expected_label": "High",
        "domain": "academic",
    },
}


def print_demo_texts() -> None:
    separator = "=" * 70
    for key, info in DEMO_TEXTS.items():
        print(separator)
        print(f"ID:       {key}")
        print(f"Domain:   {info['domain']}")
        print(f"Expected: {info['expected_label']}")
        print(f"Text:\n  {info['text'][:200]}{'…' if len(info['text']) > 200 else ''}")
    print(separator)


if __name__ == "__main__":
    print_demo_texts()

    # Optional: run through heuristic predictor if available
    try:
        from src.detection.predictor import CognitiveLoadPredictor

        predictor = CognitiveLoadPredictor(model_path="nonexistent/")
        print("\n=== Heuristic predictions ===")
        for key, info in DEMO_TEXTS.items():
            result = predictor.predict(info["text"])
            status = "✓" if result["load_label"] == info["expected_label"] else "✗"
            print(
                f"{status} {key}: predicted={result['load_label']} "
                f"(score={result['load_score']:.1f}), expected={info['expected_label']}"
            )
    except Exception as exc:
        print(f"\nCould not run predictor: {exc}")
