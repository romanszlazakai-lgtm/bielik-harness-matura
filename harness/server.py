"""Serve the harness as if it were a model, so the organisers' exam script can query it.

OpenAI style:  POST /v1/chat/completions, POST /v1/completions, GET /v1/models
Ollama style:  POST /api/chat, POST /api/generate, GET /api/tags

Usage:  python -m harness.server --port 8000
Then point the exam script at http://localhost:8000/v1 (or http://localhost:8000 for Ollama).
"""
import argparse
import json
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from .solve import solve

MODEL_ID = "matura-harness"


def _text_and_images(content):
    """Message content may be a string or a list of parts (text / image_url)."""
    if isinstance(content, str):
        return content, False
    texts, has_image = [], False
    for part in content or []:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") in ("image_url", "image", "input_image"):
            has_image = True
    return "\n".join(texts), has_image


def _question_from_messages(messages):
    """The question is the last user message; a system message may carry format rules."""
    system, user, has_image = "", "", False
    for m in messages:
        text, img = _text_and_images(m.get("content"))
        if m.get("role") == "system":
            system += text + "\n"
        elif m.get("role") == "user":
            user, has_image = text, img or bool(m.get("images"))
    # Format rules sometimes live in the system prompt; keep them next to the question.
    fmt = [l for l in system.splitlines() if any(k in l.lower() for k in ("format", "odpowiedz", "podaj"))]
    return (user + ("\n" + "\n".join(fmt) if fmt else "")).strip(), has_image


def _log(entry):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / "server_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload, status=200, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def log_message(self, fmt, *args):
        pass  # keep the console for answers only

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "team"}]})
        elif self.path.startswith("/api/tags"):
            self._send({"models": [{"name": MODEL_ID, "model": MODEL_ID}]})
        else:
            self._send({"status": "ok", "model": MODEL_ID})

    def do_POST(self):
        try:
            body = self._body()
        except json.JSONDecodeError:
            self._send({"error": "invalid json"}, status=400)
            return

        if self.path.startswith("/v1/chat/completions") or self.path.startswith("/api/chat"):
            question, has_image = _question_from_messages(body.get("messages", []))
        elif self.path.startswith("/v1/completions") or self.path.startswith("/api/generate"):
            question, has_image = body.get("prompt", ""), bool(body.get("images"))
        else:
            self._send({"error": "unknown endpoint"}, status=404)
            return

        result = solve(question, declared_type=body.get("question_type"), has_image=has_image)
        answer = result["answer"]
        print(f"[{result['type']:8}] {result['seconds']:5}s  {answer!r}  <- {question[:70]!r}")
        _log({"time": datetime.now().isoformat(timespec="seconds"), "question": question, **result})

        if self.path.startswith("/api/chat"):
            self._ollama(body, {"message": {"role": "assistant", "content": answer}})
        elif self.path.startswith("/api/generate"):
            self._ollama(body, {"response": answer})
        elif self.path.startswith("/v1/completions"):
            self._send({"id": f"cmpl-{uuid.uuid4().hex[:12]}", "object": "text_completion",
                        "created": int(time.time()), "model": MODEL_ID,
                        "choices": [{"index": 0, "text": answer, "finish_reason": "stop"}]})
        elif body.get("stream"):
            self._openai_stream(answer)
        else:
            self._send({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
                "created": int(time.time()), "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    def _ollama(self, body, part):
        base = {"model": MODEL_ID, "created_at": datetime.utcnow().isoformat() + "Z"}
        if body.get("stream", True):  # Ollama streams unless told otherwise
            lines = [dict(base, done=False, **part), dict(base, done=True, done_reason="stop",
                     **({"message": {"role": "assistant", "content": ""}} if "message" in part else {"response": ""}))]
            payload = "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines).encode("utf-8")
            self._send(payload, content_type="application/x-ndjson")
        else:
            self._send(dict(base, done=True, done_reason="stop", **part))

    def _openai_stream(self, answer):
        cid, now = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        chunks = [
            {"id": cid, "object": "chat.completion.chunk", "created": now, "model": MODEL_ID,
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]},
            {"id": cid, "object": "chat.completion.chunk", "created": now, "model": MODEL_ID,
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        payload = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self._send(payload.encode("utf-8"), content_type="text/event-stream")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from .retrieval import load_index
    load_index()  # load the knowledge base before the first question arrives
    print(f"Harness on http://{args.host}:{args.port}/v1  ->  model {config.LLM_MODEL} at {config.LLM_BASE_URL}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
