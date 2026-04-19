import json
import re
from openai import OpenAI
from typing import Any

# Use GPT-4o-mini for cheap per-chunk calls, GPT-4o for synthesis
FAST_MODEL = "gpt-4o-mini"
SMART_MODEL = "gpt-4o"

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def call_llm(prompt: str, use_smart_model: bool = False, max_tokens: int = 1500) -> str:
    """Single LLM call, returns raw text response."""
    client = get_client()
    model = SMART_MODEL if use_smart_model else FAST_MODEL

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_llm_json(prompt: str, use_smart_model: bool = False, max_tokens: int = 1500) -> Any:
    """LLM call that returns parsed JSON. Strips markdown fences if present."""
    raw = call_llm(prompt, use_smart_model=use_smart_model, max_tokens=max_tokens)

    # Strip markdown code fences if model adds them anyway
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'(\{.*\}|\[.*\])', clean, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")
