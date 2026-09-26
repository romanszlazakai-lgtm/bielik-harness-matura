"""BM25 keyword search over the local knowledge base (no extra model, works offline).

Polish is heavily inflected, so words are cut to their first 6 letters ("Grunwaldem" and
"Grunwaldu" both become "grunwa"). Years stay whole: they are the strongest keys in history.
"""
import json
import math
import pickle
import re
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
    seen = set()
    for c in candidates:
        if len(picked) >= k:
            break
        if c.get("src") == "extra" or _article(c["title"]) in seen:
            continue
        seen.add(_article(c["title"]))
        picked.append(c)
    return picked
