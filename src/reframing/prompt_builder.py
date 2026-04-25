from __future__ import annotations

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "complex": (
            "The government's implementation of austerity measures precipitated widespread "
            "socioeconomic ramifications, disproportionately affecting marginalized demographics."
        ),
        "simple": (
            "The government's spending cuts hurt many people, especially those who were "
            "already struggling financially."
        ),
    },
    {
        "complex": (
            "Photosynthesis constitutes a biochemical process whereby chlorophyll-containing "
            "organisms convert electromagnetic radiation into chemical energy stored as glucose."
        ),
        "simple": (
            "Photosynthesis is how plants use sunlight to make food. "
            "They turn light energy into sugar they can use to grow."
        ),
    },
]

SYSTEM_PROMPT = """You are a text simplification assistant. Your job is to rewrite complex text so it is easier to understand, while keeping all the facts exactly the same.

Rules you must follow:
1. Keep every fact — do not remove or change any information
2. Target a Grade 8 reading level
3. Use shorter sentences (maximum 20 words each)
4. Replace difficult words with simpler ones, but explain any word that cannot be simplified
5. Keep the same paragraph structure
6. Do not add opinions, warnings, or new information
7. Do not use bullet points unless the original does"""


def build_prompt(text: str) -> list[dict]:
    """Return a messages list for a chat model API call.

    Includes the system prompt and two few-shot examples before the target text.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples as user/assistant turns
    for example in FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": f"Please simplify this text:\n\n{example['complex']}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": example["simple"],
            }
        )

    # Add the actual request
    messages.append(
        {
            "role": "user",
            "content": f"Please simplify this text:\n\n{text}",
        }
    )

    return messages


def build_batch_prompt(texts: list[str]) -> list[list[dict]]:
    """Return a list of message lists, one per text."""
    return [build_prompt(t) for t in texts]
