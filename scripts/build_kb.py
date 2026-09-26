"""Build the local knowledge base from Polish Wikipedia (CC BY-SA) plus data/kb_extra/*.md.

Needs internet ONCE, before the exam. The exam itself runs fully offline from data/kb/.

    python scripts/build_kb.py                    # seeds + most-linked articles (default 2500)
    python scripts/build_kb.py --max-articles 800 # smaller, faster
    python scripts/build_kb.py --index-only       # rebuild chunks and index from saved articles

Steps: seed articles (data/kb_seeds.txt) -> collect their links -> keep the most linked ->
download plain-text extracts -> split into ~120-word passages -> BM25 index (data/kb/index.pkl).
"""
import argparse
import json
import pickle
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from harness import config  # noqa: E402
from harness.retrieval import BM25, CHUNKS_FILE, INDEX_FILE  # noqa: E402

API = "https://pl.wikipedia.org/w/api.php"
UA = "MaturaHarness/0.1 (Warsaw Model Trainers hackathon, educational research)"
SKIP_SECTIONS = ("Przypisy", "Bibliografia", "Linki zewnętrzne", "Zobacz też", "Uwagi", "Literatura",
                 "Galeria", "Źródła", "Filmografia", "Nagrody i odznaczenia")
_DATE_TITLE = re.compile(r"^\d+( (p\.n\.e\.|stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|"
                         r"września|października|listopada|grudnia))?$")


_last_call = [0.0]


def api(params, retries=6):
    """One request at a time, at most ~1 per second, honouring 429 Retry-After (Wikimedia etiquette)."""
    params = dict(params, format="json", formatversion=2, maxlag=5)
    body = urllib.parse.urlencode(params).encode("utf-8")
    for attempt in range(retries):
        wait = 1.0 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            req = urllib.request.Request(API, data=body, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            delay = int(exc.headers.get("Retry-After") or 0) or 5 * (attempt + 1)
            print(f"  HTTP {exc.code}, waiting {delay}s")
            time.sleep(delay)
        except Exception as exc:  # network hiccups: back off and retry
            print(f"  retry {attempt + 1}: {exc}")
            time.sleep(3 * (attempt + 1))
    return {}


def read_seeds():
    lines = (ROOT / "data" / "kb_seeds.txt").read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def links_of(title):
    out, cont = [], {}
    while True:
        data = api({"action": "query", "titles": title, "prop": "links", "plnamespace": 0,
                    "pllimit": "max", "redirects": 1, **cont})
        for page in data.get("query", {}).get("pages", []):
            out += [l["title"] for l in page.get("links", [])]
        if "continue" not in data:
            return out
        cont = data["continue"]


_NS = r"(?:Plik|File|Grafika|Image|Kategoria|Category|Media)"


def _strip_nested(text, pattern):
    """Remove innermost matches repeatedly, so nested {{templates}} disappear completely."""
    prev = None
    while prev != text:
        prev, text = text, re.sub(pattern, "", text, flags=re.S)
    return text


def clean_wikitext(text):
    """Crude wikitext -> plain text. Keeps '== headings ==' for the chunker."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = _strip_nested(text, r"\{\{[^{}]*\}\}")
    text = _strip_nested(text, r"\{\|[^{}]*?\|\}")
    prev = None
    while prev != text:  # links, innermost first, so file captions lose their inner links
        prev = text
        text = re.sub(rf"\[\[(?!{_NS}:)([^\[\]|]*)\|([^\[\]]*)\]\]", r"\2", text)
        text = re.sub(rf"\[\[(?!{_NS}:)([^\[\]|]*)\]\]", r"\1", text)
    text = re.sub(rf"\[\[{_NS}:[^\[\]]*\]\]", "", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"__[A-Z]+__", "", text)
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_batch(titles):
    """Up to 50 articles per request via revisions (extracts would allow only 1 per request)."""
    data = api({"action": "query", "titles": "|".join(titles), "prop": "revisions",
                "rvprop": "content", "rvslots": "main", "redirects": 1})
    out = []
    for page in data.get("query", {}).get("pages", []):
        revs = page.get("revisions")
        if page.get("missing") or not revs:
            continue
        content = revs[0].get("slots", {}).get("main", {}).get("content", "")
        if content.lower().startswith("#patrz") or content.lower().startswith("#redirect"):
            continue
        text = clean_wikitext(content)
        if len(text) > 200:
            out.append({"title": page["title"], "text": text})
    return out


def chunk_article(article, words_per_chunk=120):
    """Split into passages, dropping reference sections. Each passage keeps its section name."""
    chunks, section, buf = [], "", []
    skipping = False

    def flush():
        if buf:
            label = article["title"] + (f" ({section})" if section else "")
            chunks.append({"title": label, "text": " ".join(buf)})
            buf.clear()

    for para in article["text"].split("\n"):
        para = para.strip()
        if not para:
            continue
        head = re.match(r"^=+\s*(.+?)\s*=+$", para)
        if head:
            flush()
            section = head.group(1)
            skipping = section in SKIP_SECTIONS
            continue
        if skipping:
            continue
        words = para.split()
        while words:
            room = words_per_chunk - sum(len(b.split()) for b in buf)
            buf.append(" ".join(words[:room]))
            words = words[room:]
            if sum(len(b.split()) for b in buf) >= words_per_chunk:
                flush()
    flush()
    return chunks


def chunk_markdown(path):
    """Hand-written notes: one passage per '## heading' block (split further if long)."""
    text = path.read_text(encoding="utf-8")
    chunks, title = [], path.stem
    for block in re.split(r"\n(?=## )", text):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        head = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else title
        body = lines[1:] if lines[0].startswith("#") else lines
        for i in range(0, len(body), 4):
            chunks.append({"title": f"{title}: {head}", "text": " ".join(body[i:i + 4]), "src": "extra"})
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=2500)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    kb = config.KB_DIR
    kb.mkdir(parents=True, exist_ok=True)
    articles_path = kb / "articles.jsonl"

    if not args.index_only:
        seeds = read_seeds()
        links_cache = kb / "links.json"
        if links_cache.exists():
            counts = Counter(json.loads(links_cache.read_text(encoding="utf-8")))
        else:
            print(f"{len(seeds)} seed articles; collecting links (about 1 request per second)...")
            counts = Counter()
            for i, seed in enumerate(seeds, 1):
                counts.update(l for l in links_of(seed) if not _DATE_TITLE.match(l) and not l.startswith("Lista "))
                if i % 10 == 0:
                    print(f"  links {i}/{len(seeds)}")
            links_cache.write_text(json.dumps(counts, ensure_ascii=False), encoding="utf-8")
        wanted = list(dict.fromkeys(seeds + [t for t, _ in counts.most_common()]))[: args.max_articles]

        done = set()
        if articles_path.exists():
            done = {json.loads(l)["title"] for l in open(articles_path, encoding="utf-8")}
        todo = [t for t in wanted if t not in done]
        print(f"{len(wanted)} articles wanted, {len(done)} already saved, downloading {len(todo)}...")
        with open(articles_path, "a", encoding="utf-8") as out:
            for i in range(0, len(todo), 50):
                for art in fetch_batch(todo[i:i + 50]):
                    if art["title"] not in done:
                        done.add(art["title"])
                        out.write(json.dumps(art, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  {min(i + 50, len(todo))}/{len(todo)}")

    chunks = []
    for extra in sorted(config.KB_EXTRA_DIR.glob("*.md")):
        chunks += chunk_markdown(extra)
    n_extra = len(chunks)
    if articles_path.exists():
        for line in open(articles_path, encoding="utf-8"):
            chunks += chunk_article(json.loads(line))
    with open(kb / CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} passages ({n_extra} from kb_extra); building index...")

    index = BM25().build(chunks)
    with open(kb / INDEX_FILE, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = (kb / INDEX_FILE).stat().st_size / 1e6
    print(f"done: {kb / INDEX_FILE} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
