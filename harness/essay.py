"""Essay pipeline: pick a topic, draft to a fixed structure, extend until long enough, clean up.

Structure asked of the model: Wstęp (thesis + time frame) -> Argument 1-3 (fact, analysis,
conclusion) -> Podsumowanie. The length check is done in code: a small model does not count
words, so drafts under the minimum are extended with a further argument and a conclusion.
"""
import re
import time

from . import config, llm, prompts
from .retrieval import load_index

_LABEL = re.compile(r"^\s*\**\s*(Wstęp|Argument\s*\d+|Podsumowanie|Zakończenie|Teza)\s*\**\s*[:.\-]?\s*", re.I | re.M)
_SENTENCE_END = re.compile(r"[.!?…](?=\s|$)")


def word_count(text):
    return len(re.findall(r"[^\W_]+", _LABEL.sub("", text)))


def choose_topic(topics):
    """With several topics, take the one the knowledge base covers best (sum of top BM25 scores)."""
    if len(topics) == 1:
        return topics[0]
    index = load_index()
    if not index:
        return topics[0]
    return max(topics, key=lambda t: sum(h["score"] for h in index.search(t, 5)))


def essay_passages(topic, k=5):
    """More context than a closed question gets: an essay needs facts for three arguments."""
    index = load_index()
    if not index:
        return []
    seen, picked = set(), []
    for hit in index.search(topic, 40):
        article = hit["title"].split(" (")[0]
        if article in seen:
            continue
        seen.add(article)
        picked.append(dict(hit, text=hit["text"][:500]))
        if len(picked) >= k:
            break
    return picked


def trim_unfinished(text):
    """Drop a trailing half-sentence left by the token limit."""
    ends = list(_SENTENCE_END.finditer(text))
    return text[: ends[-1].end()] if ends else text


def finalize(text, topic=None):
    """Remove section labels and repeated sentences; keep one paragraph per section."""
    paragraphs, seen = [], set()
    for block in re.split(r"\n\s*\n|\n(?=\s*\**\s*(?:Wstęp|Argument\s*\d+|Podsumowanie|Zakończenie)\b)", text):
        block = _LABEL.sub("", block).strip()
        if not block:
            continue
        kept = []
        for sentence in re.split(r"(?<=[.!?…])\s+", block):
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                kept.append(sentence.strip())
        if kept:
            paragraphs.append(" ".join(kept))
    body = "\n\n".join(paragraphs)
    return f"Wybrany temat: {topic}\n\n{body}" if topic else body


def _next_argument(text):
    numbers = [int(n) for n in re.findall(r"Argument\s*(\d+)", text, re.I)]
    return max(numbers, default=3) + 1


def solve_essay(q):
    start = time.time()
    topics = q.topics or [q.text]
    topic = choose_topic(topics)
    messages = prompts.essay_messages(topic, essay_passages(topic))

    def ask(msgs):
        return llm.chat(msgs, temperature=0.3, max_tokens=config.ESSAY_MAX_TOKENS, timeout=config.ESSAY_TIMEOUT)

    raws = [ask(messages)]
    draft = trim_unfinished(raws[0].strip())
    for _ in range(config.ESSAY_MAX_EXTENSIONS):
        words = word_count(finalize(draft))  # repeated sentences are dropped later, so they do not count
        has_conclusion = re.search(r"\b(Podsumowanie|Zakończenie)\b", draft, re.I)
        if words >= config.ESSAY_MIN_WORDS and has_conclusion:
            break
        if words >= config.ESSAY_MIN_WORDS:
            request = prompts.ESSAY_CONCLUDE
        else:
            request = prompts.ESSAY_EXTEND.format(words=words, min_words=config.ESSAY_MIN_WORDS,
                                                  next_arg=_next_argument(draft))
        more = ask(messages + [{"role": "assistant", "content": draft}, {"role": "user", "content": request}])
        raws.append(more)
        if not more.strip():
            break
        draft = draft + "\n\n" + trim_unfinished(more.strip())

    final = finalize(draft, topic if len(topics) > 1 else None)
    return {
        "answer": final,
        "type": "essay",
        "topic": topic,
        "words": word_count(final),
        "extensions": len(raws) - 1,
        "votes": [],
        "raw": raws,
        "passages": [],
        "seconds": round(time.time() - start, 1),
    }
