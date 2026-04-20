import json
import re
import logging
import threading
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from typing import Any
from config import (
    FAST_MODEL, SMART_MODEL,
    LLM_MAX_RETRIES, LLM_RETRY_MIN_WAIT, LLM_RETRY_MAX_WAIT,
    COST_PER_1M
)

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()

# ─── Thread-safe cost accumulator ────────────────────────────────────────────

_cost_lock = threading.Lock()
_session_cost = {"total_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}


def get_session_cost() -> dict:
    with _cost_lock:
        return dict(_session_cost)


def reset_session_cost():
    with _cost_lock:
        _session_cost.update({"total_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0})


def _track_cost(model: str, input_tokens: int, output_tokens: int):
    pricing = COST_PER_1M.get(model, {"input": 0, "output": 0})
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    with _cost_lock:
        _session_cost["total_usd"] += cost
        _session_cost["calls"] += 1
        _session_cost["input_tokens"] += input_tokens
        _session_cost["output_tokens"] += output_tokens
    logger.debug(f"LLM cost: ${cost:.5f} ({model}, in={input_tokens}, out={output_tokens})")


# ─── Client ───────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI()
    return _client


# ─── Retry-wrapped LLM call ───────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=LLM_RETRY_MIN_WAIT, max=LLM_RETRY_MAX_WAIT),
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_llm(prompt: str, use_smart_model: bool = False, max_tokens: int = 1500) -> str:
    """Single LLM call with automatic retry on rate limits and connection errors."""
    client = get_client()
    model = SMART_MODEL if use_smart_model else FAST_MODEL

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )

    usage = response.usage
    _track_cost(model, usage.prompt_tokens, usage.completion_tokens)

    return response.choices[0].message.content


def call_llm_json(prompt: str, use_smart_model: bool = False, max_tokens: int = 1500) -> Any:
    """LLM call that returns parsed JSON. Strips markdown fences if present."""
    raw = call_llm(prompt, use_smart_model=use_smart_model, max_tokens=max_tokens)

    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        json_match = re.search(r'(\{.*\}|\[.*\])', clean, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")
