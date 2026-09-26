"""Exam-sheet text -> questions. Shared by the CKE knowledge-base builder and the exam loader.

Handles the CKE layout: 'Zadanie N.' group headers carrying the sources, 'Zadanie N.M.' subtasks,
page furniture, answer lines, inline options 'A. ... B. ...' and true/false tables marked 'PF'.
"""
import re

DASH = "[–-]"
_NOISE = [
    r"^\s*\d+(\.\d+)?\.\s*$",                      # margin task numbers "11.2."
    rf"^\s*0{DASH}\d({DASH}\d)*\s*$",              # margin points "0–1"
    r"^[\s.…_]{5,}$",                              # answer lines
    r"Strona \d+ z \d+", r"^MHIP-R0_\d+\s*$", r"arkusze\.pl", r"^BRUDNOPIS", r"nie podlega ocenie",
    r"^Egzamin maturalny z historii .{0,40}\d{4} r\.\s*$", r"^Zasady oceniania rozwiązań zadań\s*$",
    r"^Egzamin maturalny z historii\s*$", r"^Arkusz pokazowy",
    rf"^\s*\d+(\.\d+)?\.\s*0{DASH}\d\s*$",         # margin "14.1. 0–1" on one line
    r"^\s*(HISTORIA|Poziom rozszerzony|Formuła 20\d\d)\s*$", r"^\s*WYPRACOWANIE na temat",
    r"^\s*(Rozstrzygnięcie|Uzasadnienie|Odpowiedź|Nazwa|Nazwisko)\s*:\s*$",  # empty answer fields
    r"[A-ZĄĆĘŁŃÓŚŹŻ]{10,}",                         # OCR of ruled answer lines: "AAAAAAAVAAOOU"
    r"^\s*(Rozstrzygnięcie|Uzasadnienie)\s*:\s*[.e…x\s]*$",  # OCR'd empty field with its dots
]
_NOISE_RE = [re.compile(p, re.M) for p in _NOISE]
# Inline noise: OCR exports and merged forms leave page furniture mid-line.
_INLINE_NOISE = re.compile(
    r"MHIP-R0[ _]\w{3}\b|strona \d+ z \d+|Dalszy ciąg zadania na kolejnej stronie\.?"
    r"|Zadania egzaminacyjne są wydrukowane na kolejnych stronach\.?|\(\d pkt\)|\.{4,}|…{2,}", re.I)
_HEAD = re.compile(rf"(?<![\w.])Zadanie (\d+)(?:\.(\d+))?\.(?:\s*\(0{DASH}\d\))?")


def clean(text):
    lines = [l for l in text.splitlines() if not any(r.search(l) for r in _NOISE_RE)]
    text = _INLINE_NOISE.sub("", "\n".join(l.rstrip() for l in lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_sheet(text):
    """Return {task_id: {"question", "sources"}}; a group header's text becomes its subtasks' sources."""
    marks = list(_HEAD.finditer(text))
    segs = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        segs.append((m.group(1), m.group(2), clean(text[m.end():end])))
    has_sub = {num for num, sub, _ in segs if sub}
    tasks, headers = {}, {}
    # A heading can appear twice (page header, margin); keep the longest text for each id.
    for num, sub, body in segs:
        if sub:
            tid = f"{num}.{sub}"
            if len(body) >= len(tasks.get(tid, {}).get("question", "")):
                tasks[tid] = {"question": body, "sources": ""}
        elif num in has_sub:
            if len(body) >= len(headers.get(num, "")):
                headers[num] = body
        elif len(body) >= len(tasks.get(num, {}).get("question", "")):
            tasks[num] = {"question": body, "sources": ""}
    for tid, rec in tasks.items():
        if "." in tid:
            rec["sources"] = headers.get(tid.split(".")[0], "")
    return tasks


def render_tflist(body):
    """CKE true/false tables -> '(1) ... (2) ...' statements, as the harness expects them."""
    m = re.search(r"fałszywe\.", body)
    if not m:
        return None
    parts = re.split(r"\|?\s*\bP\s*\|?\s*F\b\s*\|?", body[m.end():])
    stmts = []
    for part in parts:
        part = re.sub(r"\n\s*\d\.\s+", " ", "\n" + part)
        part = re.sub(r"^\s*\d\.\s*\|?\s*|\|", " ", part.strip())
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) > 15:
            stmts.append(part)
    if len(stmts) < 2:
        return None
    lines = " ".join(f"({i}) {s}" for i, s in enumerate(stmts, 1))
    return f"{body[:m.start()].strip()} fałszywe. {lines} Odpowiedz literami P/F w kolejności."


def render_single(body):
    """Inline 'A. x B. y C. z D. w' -> stem plus one 'A) x' option per line."""
    flat = re.sub(r"\s+", " ", body)
    m = re.search(r"\bA\.\s+(.+?)\s+B\.\s+(.+?)\s+C\.\s+(.+?)\s+D\.\s+(.+?)\s*$", flat)
    if not m:
        return None
    options = "\n".join(f"{k}) {v.strip(' .')}" for k, v in zip("ABCD", m.groups()))
    return f"{flat[:m.start()].strip()}\n{options}"


def to_question_text(question, sources=""):
    """One task as the model should see it: sources first, then the normalised instruction."""
    low = question.lower()
    body = question
    if "prawdziw" in low and ("zaznacz p" in low or "p, jeśli" in low):
        body = render_tflist(question) or question
    elif "właściwą odpowiedź" in low or "dokończ zdanie" in low:
        body = render_single(question) or question
    flat_sources = re.sub(r"\s+", " ", sources)
    text = f"Źródło: {flat_sources}\n\n{body}" if sources else body
    return text.strip()


def _natural(tid):
    return [int(p) for p in tid.split(".")]


_INSTRUCTION_LINE = re.compile(
    r"^\s*(Rozstrzygnij|Podaj|Wyjaśnij|Oceń|Porównaj|Przedstaw|Scharakteryzuj|Dokończ|Zaznacz|"
    r"Uporządkuj|Przyporządkuj|Wymień|Uzasadnij|Wskaż|Napisz)\b", re.M)


def recover_headings(text):
    """Give a number to a page that holds a task instruction but no 'Zadanie N.' heading.

    CKE prints headings white on a dark bar; OCR misses them on pages that open with a picture
    (tasks 1 and 15 of the May 2023 paper). Such a page gets the next task number, so its
    instruction is not swallowed by the preamble or by the previous task.
    """
    if "\f" not in text:
        return text
    pages, last = text.split("\f"), 0
    for i, page in enumerate(pages):
        heads = list(_HEAD.finditer(page))
        if heads:
            last = int(heads[-1].group(1))
        elif _INSTRUCTION_LINE.search(page) and len(re.findall(r"[^\W\d_]", page)) > 40:
            last += 1
            pages[i] = f"Zadanie {last}.\n{page}"
    return "\f".join(pages)


def questions_from_text(text):
    """Split a whole sheet into dev/exam items: [{"id", "question"}] in task order."""
    tasks = parse_sheet(recover_headings(text))
    if tasks:
        return [{"id": f"Z{tid}", "question": to_question_text(rec["question"], rec["sources"])}
                for tid, rec in sorted(tasks.items(), key=lambda kv: _natural(kv[0])) if rec["question"]]
    # No "Zadanie N." headings: fall back to numbered paragraphs, else blank-line blocks.
    blocks = re.split(r"\n(?=\s*\d{1,2}[.)]\s)", clean(text)) if re.search(r"\n\s*\d{1,2}[.)]\s", text) \
        else re.split(r"\n\s*\n", clean(text))
    return [{"id": f"Q{i}", "question": b.strip()} for i, b in enumerate(blocks, 1) if len(b.strip()) > 30]
