"""Question parsing: detect the answer type and pull out options, statements and elements.

Types follow the organisers' exam sheet (see data/dev/README.md):
single, multi, tflist, matching, order, numeric, short, open.
"""
import re
from dataclasses import dataclass, field

TYPES = ("single", "multi", "tflist", "matching", "order", "numeric", "short", "open")
LETTERS = "ABCDEFGH"

# Letters used as option labels: "A) text", "A. text", "A: text" at line start.
_OPT_LINE = re.compile(r"^\s*([A-H])\s*[\).:]\s+(.+?)\s*$", re.M)
# Inline labels: "A) text; B) text".
_OPT_INLINE = re.compile(r"(?<![0-9A-Za-zÀ-ſ])([A-H])\)\s*")
_INSTRUCTION = re.compile(
    r"\.\s+(?=(Podaj|Odpowiedz|Wpisz|Zaznacz|Uporządkuj|Wybierz)\b)|\s+(?=(Podaj|Odpowiedz)\s)"
)
_STATEMENT = re.compile(r"\((\d+)\)\s*(.+?)(?=\(\d+\)|Odpowiedz|$)", re.S)


@dataclass
class Question:
    text: str
    qtype: str
    options: dict = field(default_factory=dict)      # letter -> option text
    elements: list = field(default_factory=list)     # matching: element names in order
    statements: list = field(default_factory=list)   # tflist: statements in order
    has_image: bool = False

    @property
    def letters(self):
        return sorted(self.options) or list("ABCD")

    @property
    def stem(self):
        """The question without its answer options (options add noise to retrieval)."""
        return re.split(r"\n\s*A\s*[\).:]\s|\sA\)\s", self.text, maxsplit=1)[0]


def _sequential(letters):
    return letters == list(LETTERS[: len(letters)])


def parse_options(text):
    """Return {letter: text}. Tries one-per-line labels first, then inline labels."""
    lines = _OPT_LINE.findall(text)
    if len(lines) >= 2 and _sequential([l for l, _ in lines]):
        return {l: t.strip().rstrip(";") for l, t in lines}

    marks = list(_OPT_INLINE.finditer(text))
    if len(marks) >= 2 and _sequential([m.group(1) for m in marks]):
        opts = {}
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            seg = text[m.end():end]
            if i + 1 == len(marks):
                seg = _INSTRUCTION.split(seg)[0]
            opts[m.group(1)] = seg.strip().rstrip(";.").strip()
        return opts
    return {}


def parse_elements(text):
    """Matching elements, taken from the format example 'np. granit=?; bazalt=?'."""
    for m in re.finditer(r"np\.\s*((?:[^;=\n]+=\s*\?\s*;?\s*)+)", text):
        found = [e.strip(" .,;") for e in re.findall(r"([^;=\n]+?)\s*=\s*\?", m.group(1))]
        if found:
            return found
    m = re.search(r"Elementy:\s*(.+?)\.\s", text + " ")
    if m:
        return [e.strip() for e in m.group(1).split(",") if e.strip()]
    return []


def parse_statements(text):
    stmts = [s.strip() for _, s in _STATEMENT.findall(text)]
    if stmts:
        return stmts
    return [l.strip() for l in re.findall(r"^\s*\d+[.)]\s*(.+)$", text, re.M)]


def detect_type(text, has_options):
    low = text.lower()
    if "samą liczbę" in low or "sama liczbe" in low or re.search(r"\boblicz\b", low):
        return "numeric"
    if "p/f" in low or "(prawda) lub f" in low or "prawda/fałsz" in low or "prawdziwość" in low:
        return "tflist"
    if "element=litera" in low or "=?" in text:
        return "matching"
    if low.lstrip().startswith("uporządkuj") or ("w kolejności" in low and "litery" in low):
        return "order"
    if any(k in low for k in ("kilka poprawnych", "wszystkie litery", "zaznacz wszystkie",
                              "wszystkie poprawne", "wszystkie odpowiedzi")):
        return "multi"
    if has_options:
        return "single"
    if any(k in low for k in ("wypracowanie", "uzasadnij", "wyjaśnij", "porównaj", "oceń, czy")):
        return "open"
    return "short"


def format_question(item):
    """Question text as sent to the model: stem, then one option per line.

    Accepts options as {"A": "..."} or ["...", "..."], under "options" or "choices".
    """
    text = str(item.get("question") or item.get("prompt") or item.get("text") or "").strip()
    options = item.get("options") or item.get("choices")
    if isinstance(options, list):
        options = {LETTERS[i]: str(o) for i, o in enumerate(options)}
    if options:
        text += "\n" + "\n".join(f"{k}) {v}" for k, v in options.items())
    return text


def parse(text, declared_type=None, has_image=False):
    options = parse_options(text)
    qtype = declared_type if declared_type in TYPES else detect_type(text, bool(options))
    q = Question(text=text.strip(), qtype=qtype, options=options, has_image=has_image)
    if qtype == "matching":
        q.elements = parse_elements(text)
        m = re.search(r"Kategorie:\s*(.+)", text, re.S)
        if m:
            q.options = parse_options(m.group(1)) or options
    elif qtype == "tflist":
        q.statements = parse_statements(text)
    return q
