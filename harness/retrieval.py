"""BM25 keyword search over the local knowledge base (no extra model, works offline).

Polish is heavily inflected, so words are cut to their first 6 letters ("Grunwaldem" and
"Grunwaldu" both become "grunwa"). Years stay whole: they are the strongest keys in history.
"""
import json
import math
import re
import pickle
from array import array
from collections import Counter, defaultdict

from . import config

_TOKEN = re.compile(r"\d+|[^\W\d_]+", re.UNICODE)
_STOP = set("""
a aby ale bo by był była było były być co czy dla do go i ich ile im in iż jak jako je jego jej jest
jeszcze już każdy kiedy kto która które którego której który którzy ku lub ma mu na nad nie niż o od
oraz po pod podaj przed przez przy r roku się są ta tak tam te tego tej ten to tu tym u w we według wśród
z za ze że żeby oceń odpowiedz litery literę poprawne poprawnych zaznacz wszystkie np
""".split())

INDEX_FILE = "index.pkl"
CHUNKS_FILE = "chunks.jsonl"


def tokenize(text):
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        if tok.isdigit():
            out.append(tok)
        elif len(tok) > 1:
            out.append(tok[:6])
    return out


class BM25:
    def __init__(self, k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.docs = []          # list of {"title", "text"}
        self.lengths = array("I")
        self.postings = {}      # term -> (array of doc ids, array of term frequencies)
        self.idf = {}
        self.avgdl = 1.0

    def build(self, docs):
        tmp = defaultdict(lambda: (array("I"), array("H")))
        for i, doc in enumerate(docs):
            toks = tokenize(doc["title"] + " " + doc["text"])
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                ids, tfs = tmp[term]
                ids.append(i)
                tfs.append(min(tf, 65535))
        self.docs = docs
        self.postings = dict(tmp)
        n = max(len(docs), 1)
        self.avgdl = (sum(self.lengths) / n) or 1.0
        self.idf = {t: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
                    for t, (ids, _) in self.postings.items()}
        return self

    def search(self, query, k=3):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tfs = self.postings[term]
            idf = self.idf[term]
            for doc_id, tf in zip(ids, tfs):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [dict(self.docs[i], score=round(s, 2)) for i, s in best]


_INDEX = None


def load_index():
    """Load the pickled index once; build it from chunks.jsonl if the pickle is missing."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    pkl = config.KB_DIR / INDEX_FILE
    if pkl.exists():
        with open(pkl, "rb") as f:
            _INDEX = pickle.load(f)
        return _INDEX
    chunks = config.KB_DIR / CHUNKS_FILE
    if not chunks.exists():
        print(f"[rag] no knowledge base at {chunks}; answering without retrieval")
        _INDEX = False
        return _INDEX
    with open(chunks, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    _INDEX = BM25().build(docs)
    with open(pkl, "wb") as f:
        pickle.dump(_INDEX, f, protocol=pickle.HIGHEST_PROTOCOL)
    return _INDEX


def _article(title):
    return title.split(" (")[0]


_CKE_INDEX = None
_BOILERPLATE = re.compile(
    r"\b(podaj|stosowan\w*|historiografii|nazw\w*|źródł\w*|fragment\w*|opracowani\w*|historyczn\w*|tekst\w*"
    r"|zadani\w*|odpowied\w*|zaznacz\w*|właściw\w*|spośród|podanych|dokończ|zdanie|oceń|prawdziwoś\w*"
    r"|stwierdze\w*|wpisz|prawda|fałsz|kolejności|jak|najkrócej|literami|wspomnian\w*|opisan\w*"
    r"|przedstawion\w*|ilustracj\w*|mapie|mapa|na podstawie|dotyczy|mowa)\b", re.I)


def _coverage(query_tokens, doc):
    """Share of the query's words that also occur in the passage."""
    return len(query_tokens & set(tokenize(doc["title"] + " " + doc["text"]))) / max(len(query_tokens), 1)


def cke_passage(query):
    """The marking-scheme answer for a question that repeats a CKE task, or None.

    Only near-duplicates count: most of the question's words must occur in one passage. When the
    question matches a task's sources, the answer passage of that task (best-matching subtask) is
    returned instead, since the sources alone do not contain the answer.
    """
    global _CKE_INDEX
    if _CKE_INDEX is None:
        path = config.CKE_DIR / INDEX_FILE
        _CKE_INDEX = False
        if path.exists():
            with open(path, "rb") as f:
                _CKE_INDEX = pickle.load(f)
        else:
            print(f"[rag] USE_CKE=1 but {path} is missing; run scripts/build_cke.py")
    if not _CKE_INDEX:
        return None
    # Exam boilerplate matches every CKE task equally; only the historical content should count.
    query = _BOILERPLATE.sub(" ", query)
    wanted = set(tokenize(query))
    hits = _CKE_INDEX.search(query, 8)
    best = next((h for h in hits if _coverage(wanted, h) >= config.CKE_MIN_COVERAGE), None)
    m = best and re.match(r"CKE (\S+) zad\. (\d+)", best["title"])
    if not m:
        return None
    # Candidates: every answer passage of that task group; the one whose own instruction the
    # question repeats wins. Short generic questions fail this test, which stops false alarms.
    exam, group = m.group(1), m.group(2)
    answers = [d for d in _CKE_INDEX.docs
               if re.fullmatch(rf"CKE {re.escape(exam)} zad\. {group}(\.\d+)?", d["title"])]
    scored = [(_instruction_overlap(wanted, d), d) for d in answers]
    overlap, pick = max(scored, key=lambda s: s[0], default=(0, None))
    return pick if overlap >= config.CKE_MIN_INSTRUCTION else None


def _instruction_overlap(query_tokens, doc):
    """Share of the CKE task's own instruction words that the question contains."""
    instruction = doc["text"].split(" | Odpowiedź", 1)[0].replace("Polecenie:", "")
    words = set(tokenize(_BOILERPLATE.sub(" ", instruction)))
    return len(words & query_tokens) / max(len(words), 1)


def retrieve(q, k=None):
    """Top passages: the best cheat-sheet passage first, then Wikipedia, one passage per article."""
    if not config.USE_RAG:
        return []
    index = load_index()
    if not index:
        return []
    k = k or config.TOP_K
    query = q.stem if q.qtype in ("single", "multi", "short") else q.text
    candidates = index.search(query, 40)
    picked = [c for c in candidates if c.get("src") == "extra"][:1]
    if config.USE_CKE:
        # A matching CKE task or source takes the first Wikipedia slot; the prompt stays the same size.
        cke = cke_passage(q.text)  # full text: the options are part of what identifies a task
        if cke:
            picked.insert(0, cke)
    seen = set()
    for c in candidates:
        if len(picked) >= k:
            break
        if c.get("src") == "extra" or _article(c["title"]) in seen:
            continue
        seen.add(_article(c["title"]))
        picked.append(c)
    return picked
