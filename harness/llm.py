"""Minimal OpenAI-compatible chat client (standard library only)."""
import json
import re
import urllib.error
import urllib.request

from . import config

_THINK = re.compile(r"<think>.*?</think>", re.S)


def chat(messages, temperature=0.0, max_tokens=None, base_url=None, model=None):
    """Send one chat request and return the assistant text ("" on failure)."""
    body = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or config.MAX_TOKENS,
        "stream": False,
    }
    url = (base_url or config.LLM_BASE_URL).rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[llm] request failed: {exc}")
        return ""
    text = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
    return _THINK.sub("", text).strip()
