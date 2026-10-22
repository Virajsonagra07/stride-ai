"""Thin Groq wrapper. Every call degrades gracefully: if no key / API error,
callers get None and fall back to the deterministic path, so the demo never dies on stage."""
import json
import re
import time

from . import config

_client = None


def _key() -> str:
    # Read lazily so a key added in Streamlit Secrets after startup is picked up without a restart.
    return config._secret("GROQ_API_KEY") or config.GROQ_API_KEY


def available() -> bool:
    k = _key()
    return bool(k) and "your_key" not in k


def _get():
    global _client
    if _client is None and available():
        from groq import Groq
        _client = Groq(api_key=_key())
    return _client


def chat(messages, model=None, temperature=0.2, max_tokens=700, json_mode=False):
    """Returns (text, latency_ms) or (None, 0) on failure."""
    client = _get()
    if client is None:
        return None, 0
    kwargs = dict(model=model or config.GROQ_MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content, int((time.perf_counter() - t0) * 1000)
    except Exception as e:  # network, rate limit, retired model ...
        print(f"[llm] {type(e).__name__}: {e}")
        return None, 0


def chat_json(messages, model=None, retries=1):
    """JSON-mode call with one retry (small models occasionally return malformed JSON)."""
    total = 0
    for _ in range(retries + 1):
        text, ms = chat(messages, model=model or config.GROQ_FAST_MODEL, temperature=0,
                        max_tokens=600, json_mode=True)
        total += ms
        if not text:
            continue
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0) if m else text), total
        except json.JSONDecodeError:
            continue
    return None, total
