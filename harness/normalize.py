"""Turn free model text into exactly the answer format the exam grader expects.

Every function returns a string in canonical form, or "" when nothing usable was found.
"""
import ast
import operator
import re

from . import config

_WORD = "A-Za-zÀ-ſ"
_LETTER = re.compile(rf"(?<![{_WORD}])([A-H])(?![{_WORD}])")
_ANSWER_MARK = re.compile(r"odpowied[zź]\s*(?:to|brzmi)?\s*[:=]", re.I)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_TF_TOKEN = re.compile(
    rf"(?<![{_WORD}])(P|F|prawda|prawdziwe|fałsz|fałszywe|falsz|tak|nie)(?![{_WORD}])", re.I
)


def answer_tail(raw):
    """Text after the last 'Odpowiedź:' marker, or the whole text if there is none."""
    marks = list(_ANSWER_MARK.finditer(raw))
    return raw[marks[-1].end():].strip() if marks else ""


def _last_line(raw):
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _letters_in(text, allowed):
    return [l for l in _LETTER.findall(text) if l in allowed]


def single(q, raw):
    allowed = q.letters
    tail = answer_tail(raw)
    if tail:
        found = _letters_in(tail, allowed)
        if found:
            return found[0]
    found = _letters_in(_last_line(raw), allowed)
    if found:
        return found[-1]
    # The model may have written the option text instead of its letter.
    low = raw.lower()
    best, pos = "", -1
    for letter, text in q.options.items():
        p = low.rfind(text.lower()) if text else -1
        if p > pos:
            best, pos = letter, p
    if best:
        return best
    found = _letters_in(raw, allowed)
    return found[-1] if found else ""


def multi(q, raw):
    allowed = q.letters
    for chunk in (answer_tail(raw), _last_line(raw)):
        found = sorted(set(_letters_in(chunk, allowed)))
        if found:
            return ",".join(found)
    return ""


def tflist(q, raw):
    n = len(q.statements)
    for chunk in (answer_tail(raw), _last_line(raw), raw):
        tokens = []
        for t in _TF_TOKEN.findall(chunk):
            t = t.lower()
            tokens.append("P" if t in ("p", "prawda", "prawdziwe", "tak") else "F")
        if tokens and (n == 0 or len(tokens) >= n):
            tokens = tokens[:n] if n else tokens
            return ",".join(tokens)
        if tokens and chunk is raw:
            return ",".join(tokens + ["P"] * (n - len(tokens)))
    return ""


def matching(q, raw):
    allowed = q.letters
    tail = answer_tail(raw) or raw
    result = {}
    for element in q.elements:
        pattern = re.escape(element) + r"\s*(?:=|:|->|→|-|–)\s*\(?([A-H])(?![" + _WORD + "])"
        for chunk in (tail, raw):
            m = re.search(pattern, chunk, re.I)
            if m and m.group(1) in allowed:
                result[element] = m.group(1)
                break
    if len(result) < len(q.elements):
        # Fall back to position: "x=A; y=B" pairs in the order the model wrote them.
        pairs = re.findall(rf"([^;=\n,]+?)\s*=\s*([A-H])(?![{_WORD}])", tail)
        if len(pairs) == len(q.elements):
            for element, (_, letter) in zip(q.elements, pairs):
                result.setdefault(element, letter)
    if not result:
        return ""
    unused = [l for l in allowed if l not in result.values()]
    for element in q.elements:
        if element not in result:
            result[element] = unused.pop(0) if unused else allowed[0]
    return "; ".join(f"{e}={result[e]}" for e in q.elements)


def order(q, raw):
    allowed = q.letters
    for chunk in (answer_tail(raw), _last_line(raw)):
        seq = []
        for l in _letters_in(chunk, allowed):
            if l not in seq:
                seq.append(l)
        if len(seq) >= 2:
            seq += [l for l in allowed if l not in seq]
            return ",".join(seq)
    return ""


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_eval(expr):
    """Evaluate plain arithmetic only (numbers, + - * / ** and brackets)."""
    expr = expr.replace(",", ".").replace("×", "*").replace("·", "*").replace(":", "/")
    expr = re.sub(r"[^0-9.+\-*/() ]", "", expr)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr, mode="eval"))


def format_number(value):
    value = round(float(value), 2)
    text = str(int(value)) if value == int(value) else f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", config.DECIMAL_SEPARATOR)


def numeric(q, raw):
    m = re.search(r"WYRAŻENIE\s*:\s*(.+)", raw, re.I)
    if m:
        try:
            return format_number(safe_eval(m.group(1).split("=")[0]))
        except (ValueError, SyntaxError, ZeroDivisionError):
            pass
    for chunk, pick in ((answer_tail(raw), 0), (_last_line(raw), -1), (raw, -1)):
        nums = _NUMBER.findall(chunk)
        if nums:
            return format_number(nums[pick].replace(",", "."))
    return ""


def short(q, raw):
    text = answer_tail(raw) or _last_line(raw) or raw
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"[*_\"'„”«»]", "", text).strip(" .;:")
    words = text.split()
    return " ".join(words[:8])


def open_answer(q, raw):
    return (answer_tail(raw) or raw).strip()


NORMALIZERS = {
    "single": single, "multi": multi, "tflist": tflist, "matching": matching,
    "order": order, "numeric": numeric, "short": short, "open": open_answer,
}


def normalize(q, raw):
    return NORMALIZERS[q.qtype](q, raw or "")
