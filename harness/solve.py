"""The harness pipeline: parse -> retrieve -> prompt -> sample votes -> normalize -> vote."""
import re
import time
from collections import Counter

from . import config, llm, normalize, prompts, qtypes, retrieval


def _majority(values):
    """Most common value; ties go to the earliest (the greedy, temperature-0 answer)."""
    values = [v for v in values if v]
    if not values:
        return ""
    counts = Counter(values)
    top = max(counts.values())
    return next(v for v in values if counts[v] == top)


def _vote_positions(answers, sep):
    """Per-position majority for 'P,F,P' style answers."""
    rows = [a.split(sep) for a in answers if a]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    out = []
    for i in range(width):
        out.append(_majority([r[i].strip() for r in rows if i < len(r)]))
    return sep.join(out)


def _vote_multi(answers):
    rows = [set(a.split(",")) for a in answers if a]
    if not rows:
        return ""
    counts = Counter(l for r in rows for l in r)
    chosen = sorted(l for l, c in counts.items() if c * 2 > len(rows))
    return ",".join(chosen) if chosen else answers[0]


def _vote_matching(answers):
    rows = [dict(p.split("=", 1) for p in a.split("; ")) for a in answers if a]
    if not rows:
        return ""
    keys = list(rows[0])
    return "; ".join(f"{k}={_majority([r.get(k, '') for r in rows])}" for k in keys)


def combine(qtype, answers):
    if qtype == "tflist":
        return _vote_positions(answers, ",")
    if qtype == "multi":
        return _vote_multi(answers)
    if qtype == "matching":
        return _vote_matching(answers)
    return _majority(answers)


def solve(text, declared_type=None, has_image=False, votes=None):
    """Answer one exam question. Returns a dict with the final answer and a trace."""
    start = time.time()
    q = qtypes.parse(text, declared_type=declared_type, has_image=has_image)
    if q.qtype == "essay":
        from .essay import solve_essay
        return solve_essay(q)
    passages = retrieval.retrieve(q)
    messages = prompts.build_messages(q, passages)

    n_votes = votes or config.VOTES
    if q.qtype == "open":
        n_votes = 1
    raws, answers = [], []
    for i in range(n_votes):
        temperature = 0.0 if i == 0 else config.VOTE_TEMPERATURE
        raw = llm.chat(messages, temperature=temperature, max_tokens=prompts.max_tokens(q))
        raws.append(raw)
        answers.append(normalize.normalize(q, raw))
        # Stop early when the first two samples already agree.
        if i == 1 and answers[0] and answers[0] == answers[1]:
            break

    final = combine(q.qtype, answers) or _fallback(q)
    # With nothing to go on (a picture it cannot see), the model copies the worked example
    # ("Unia w Krewie" for a question about a Neolithic village). Ask again without the example.
    if echoes_example(final, messages, q.text):
        raw = llm.chat([messages[0], messages[-1]], temperature=0.0, max_tokens=prompts.max_tokens(q))
        raws.append(raw)
        final = normalize.normalize(q, raw) or final
    return {
        "answer": final,
        "type": q.qtype,
        "votes": answers,
        "raw": raws,
        "passages": [p["title"] for p in passages],
        "seconds": round(time.time() - start, 1),
    }


def _stems(text):
    return {w[:6] for w in re.findall(r"[^\W\d_]{4,}", text.lower())}


def echoes_example(answer, messages, question):
    """True when most of the answer's words come from the worked example and not from the question."""
    if len(messages) < 4 or not answer or len(answer) < 3:
        return False
    if messages[-2]["role"] != "assistant":
        return False
    # The whole worked example counts: the model also copies the example's question wording.
    example = _stems(messages[-3]["content"]) | _stems(messages[-2]["content"])
    first = re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)[0]
    for part in (answer, first):
        words = _stems(part)
        if words and len((words & example) - _stems(question)) >= max(1, len(words) / 2):
            return True
    return False


def _fallback(q):
    """Never return an empty answer for a closed question: a guess scores more than a blank."""
    if q.qtype in ("single", "multi"):
        return q.letters[0]
    if q.qtype == "order":
        return ",".join(q.letters)
    if q.qtype == "tflist":
        return ",".join(["P"] * max(len(q.statements), 1))
    if q.qtype == "matching" and q.elements:
        return "; ".join(f"{e}={q.letters[i % len(q.letters)]}" for i, e in enumerate(q.elements))
    return ""
