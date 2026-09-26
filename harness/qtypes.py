"""Question parsing: detect the answer type and pull out options, statements and elements.

Types follow the organisers' exam sheet (see data/dev/README.md):
single, multi, tflist, matching, order, numeric, short, open, essay.
"""
import re
from dataclasses import dataclass, field

TYPES = ("single", "multi", "tflist", "matching", "order", "numeric", "short", "open", "essay")
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
    topics: list = field(default_factory=list)       # essay: topics to choose from
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


_ESSAY = re.compile(r"wypracowani|rozprawk|w formie (?:wypowiedzi|pracy|eseju)|\besej|co najmniej \d+ sł|"
                    r"wybierz jeden z (?:trzech |podanych )?temat|sformułuj (?:i uzasadnij )?stanowisko")
_OPEN_VERBS = re.compile(r"\b(wyjaśnij|uzasadnij|rozstrzygnij|porównaj|scharakteryzuj|przedstaw|opisz|"
                         r"omów|oceń|wykaż|udowodnij|zinterpretuj|wskaż i uzasadnij)\b")
_SHORT_MARKS = re.compile(r"jak najkrócej|jednym słowem|samą nazwę|samo nazwisko")
TYPE_ALIASES = {"wypracowanie": "essay", "long": "essay", "essay_question": "essay",
                "text": "open", "opisowe": "open", "otwarte": "open", "explain": "open"}


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
    # Before "single": essay topics are often labelled A) B) C) like options.
    if _ESSAY.search(low):
        return "essay"
    if has_options:
        return "single"
    # An explicit "as briefly as possible" wins; otherwise any verb asking for reasons means open.
    if _SHORT_MARKS.search(low):
        return "short"
    if _OPEN_VERBS.search(low):
        return "open"
    return "short"


def parse_topics(text, options):
    """Essay topics: 'A) ...' labels, 'Temat 1: ...' or numbered lines; else the whole prompt."""
    if len(options) >= 2:
        return list(options.values())
    found = re.findall(r"(?:^|\n)\s*(?:Temat\s*)?\d[.):]\s*(.{15,}?)(?=\n\s*(?:Temat\s*)?\d[.):]|\Z)", text, re.S)
    topics = [re.sub(r"\s+", " ", t).strip() for t in found]
    return topics if len(topics) >= 2 else [text.strip()]


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
    declared_type = TYPE_ALIASES.get(declared_type, declared_type)
    qtype = declared_type if declared_type in TYPES else detect_type(text, bool(options))
    q = Question(text=text.strip(), qtype=qtype, options=options, has_image=has_image)
    if qtype == "essay":
        q.topics = parse_topics(text, options)
        q.options = {}
    elif qtype == "matching":
        q.elements = parse_elements(text)
        m = re.search(r"Kategorie:\s*(.+)", text, re.S)
        if m:
            q.options = parse_options(m.group(1)) or options
    elif qtype == "tflist":
        q.statements = parse_statements(text)
    return q
