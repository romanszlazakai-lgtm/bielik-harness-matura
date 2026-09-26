"""Essay pipeline: pick a topic, draft to a fixed structure, extend until long enough, clean up.

Structure asked of the model: Wstęp (thesis + time frame) -> Argument 1-3 (fact, analysis,
conclusion) -> Podsumowanie. The length check is done in code: a small model does not count
words, so drafts under the minimum are extended with a further argument and a conclusion.
"""
import re
import time

from . import config, llm, prompts
from .retrieval import load_index

_TOPIC_WORDING = re.compile(
    r"\b(przyczyn\w*|skutk\w*|podobn\w*|oceń|wpływ\w*|uwzględni\w*|aspekt\w*|polityczn\w*|społeczn\w*|"
    r"gospodarcz\w*|kulturow\w*|stanowisk\w*|tez\w*|uzasadnij|zajmij|wobec|powyższej|scharakteryzuj|"
    r"porównaj|wybran\w*|trzech|argumentacji|swojej|charakteryzując|miały|miał|były|był|napisz|"
    r"wypracowani\w*|temat\w*|stwierdzeni\w*|czynnik\w*|tego|okresu|powinno|liczyć|słów|najmniej)\b", re.I)
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


def essay_passages(topic, k=5, timeline=2):
    """Context for three arguments: the best cheat-sheet (timeline) passages first, then Wikipedia.

    The timeline carries the dates and names an essay needs; without it the 1.5B model invented
    them (the first test essay dated the constitutional monarchy to 1790 and the KEN to 1790).
    """
    index = load_index()
    if not index:
        return []
    # Task wording ("przyczyny", "oceń", "uwzględnij aspekty") matches hundreds of articles; the
    # search should run on the historical content of the topic only.
    query = _TOPIC_WORDING.sub(" ", topic)
    # The timeline is ranked on its own: among 56 000 Wikipedia passages it rarely reaches the top.
    everything = index.search(query, len(index.docs))
    picked = [dict(h, text=h["text"][:500]) for h in everything if h.get("src") == "extra"][:timeline]
    hits = everything[:60]
    seen = set()
    for hit in hits:
        if len(picked) >= k:
            break
        article = hit["title"].split(" (")[0]
        if hit.get("src") == "extra" or article in seen:
            continue
        seen.add(article)
        picked.append(dict(hit, text=hit["text"][:500]))
    return picked


def trim_unfinished(text):
    """Drop a trailing half-sentence left by the token limit."""
    ends = list(_SENTENCE_END.finditer(text))
    return text[: ends[-1].end()] if ends else text


_HEADING = re.compile(r"^\s*(#+\s|Temat\s*:|Wypracowanie\s*:)", re.I)   # whole line dropped
_INLINE_LABEL = re.compile(r"^\s*(Wnioski|Wniosek|Analiza|Fakt(?: historyczny)?|Teza|Stanowisko)\s*:\s*", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")


def _stem_set(sentence):
    return {w[:6] for w in re.findall(r"[^\W\d_]{3,}", sentence.lower())}


def _prose(block):
    """Markdown and bullet lists -> plain sentences; headings without a sentence are dropped."""
    lines = []
    for line in block.splitlines():
        line = re.sub(r"\*\*|__|`", "", line).strip()
        if not line or _HEADING.match(line):
            continue
        is_bullet = bool(_BULLET.match(line))
        line = _BULLET.sub("", line)
        if line.endswith(":"):  # "- Czynniki polityczne:" introduces a list, it is not a sentence
            continue
        # "Wniosek: ...", "Fakt historyczny: Deklaracja (1776)": the label goes, the content stays.
        labelled = bool(_INLINE_LABEL.match(line))
        line = _INLINE_LABEL.sub("", line)
        # "Konflikt z Wielką Brytanią: Koloniści..." keeps the sentence, loses the short label.
        m = re.match(r"^([^.:!?]{2,60}):\s+(.+)$", line)
        if m and len(m.group(1).split()) <= 6:
            line, labelled = m.group(2), True
        # A short line with no sentence end and no label is a heading ("Reforma gospodarcza").
        if not re.search(r"[.!?…]$", line) and len(line.split()) <= 8 and not (is_bullet or labelled):
            continue
        if not line:
            continue
        if not re.search(r"[.!?…]$", line):
            line += "."
        lines.append(line[0].upper() + line[1:])
    return " ".join(lines)


def finalize(text, topic=None, similarity=0.7):
    """Plain prose, one paragraph per section; repeated and near-repeated sentences are removed."""
    paragraphs, kept_stems = [], []
    for block in re.split(r"\n\s*\n|\n(?=\s*\**\s*(?:Wstęp|Argument\s*\d+|Podsumowanie|Zakończenie)\b)", text):
        block = _prose(_LABEL.sub("", block.strip()))
        if not block:
            continue
        kept = []
        for sentence in re.split(r"(?<=[.!?…])\s+", block):
            stems = _stem_set(sentence)
            if not stems:
                continue
            # Near-duplicate: most words shared with a sentence already kept ("kolonialnych koloniach").
            if any(len(stems & old) / max(len(stems | old), 1) >= similarity for old in kept_stems):
                continue
            kept_stems.append(stems)
            kept.append(sentence.strip())
        if kept:
            paragraphs.append(" ".join(kept))
    body = "\n\n".join(paragraphs)
    return f"Wybrany temat: {topic}\n\n{body}" if topic else body


def _next_argument(text):
    numbers = [int(n) for n in re.findall(r"Argument\s*(\d+)", text, re.I)]
    return max(numbers, default=3) + 1


_CONCLUSION = re.compile(r"(?:^|\n)\s*[*#\s]*(Podsumowanie|Zakończenie)\b", re.I)


def split_conclusion(draft):
    """(body, conclusion); the conclusion starts at its label, or is empty when there is none."""
    m = _CONCLUSION.search(draft)
    return (draft[:m.start()].rstrip(), draft[m.start():].strip()) if m else (draft, "")


def solve_essay(q):
    start = time.time()
    topics = q.topics or [q.text]
    topic = choose_topic(topics)
    messages = prompts.essay_messages(topic, essay_passages(topic))

    def ask(msgs):
        # Essays may go to their own model and server (ESSAY_LLM_MODEL / ESSAY_LLM_BASE_URL).
        return llm.chat(msgs, temperature=0.3, max_tokens=config.ESSAY_MAX_TOKENS, timeout=config.ESSAY_TIMEOUT,
                        base_url=config.ESSAY_LLM_BASE_URL or None, model=config.ESSAY_LLM_MODEL or None)

    raws = [ask(messages)]
    draft = trim_unfinished(raws[0].strip())
    for _ in range(config.ESSAY_MAX_EXTENSIONS):
        words = word_count(finalize(draft))  # repeated sentences are dropped later, so they do not count
        body, conclusion = split_conclusion(draft)
        if words >= config.ESSAY_MIN_WORDS and conclusion:
            break
        need = max(config.ESSAY_MIN_WORDS - words + 40, 80)
        fmt = dict(words=words, min_words=config.ESSAY_MIN_WORDS, need=need, next_arg=_next_argument(draft))
        if words >= config.ESSAY_MIN_WORDS:
            request = prompts.ESSAY_CONCLUDE
        elif conclusion:
            request = prompts.ESSAY_EXTEND_ARGUMENT.format(**fmt)
        else:
            request = prompts.ESSAY_EXTEND.format(**fmt)
        more = ask(messages + [{"role": "assistant", "content": draft}, {"role": "user", "content": request}])
        raws.append(more)
        if not more.strip():
            break
        addition = trim_unfinished(more.strip())
        if conclusion and words < config.ESSAY_MIN_WORDS:
            # A new argument goes before the existing conclusion, never after it.
            draft = f"{body}\n\n{split_conclusion(addition)[0]}\n\n{conclusion}"
        else:
            draft = f"{draft}\n\n{addition}"

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
