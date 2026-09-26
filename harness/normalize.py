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
    # Preferred: one verdict per option ("A: TAK", "B: NIE") written before the answer line.
    verdicts = dict(re.findall(rf"(?<![{_WORD}])([A-H])\s*[:)\-]\s*(TAK|NIE)\b", _reasoning(raw), re.I))
    if len(verdicts) >= max(2, len(allowed) - 1):
        chosen = sorted(l for l, v in verdicts.items() if v.upper() == "TAK" and l in allowed)
        if chosen:
            return ",".join(chosen)
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


def _reasoning(raw):
    """Model text before the final 'Odpowiedź:' line."""
    marks = list(_ANSWER_MARK.finditer(raw))
    return raw[:marks[-1].start()] if marks else raw


def _stems(text):
    return {t if t.isdigit() else t[:6] for t in re.findall(r"\d+|[^\W\d_]{3,}", text.lower())}


def _match_by_content(q, raw):
    """Map each element to the category whose text the model wrote next to it.

    Small models often recall the right fact ('Grunwald 1410') but pick the wrong letter;
    matching on the category text instead of the letter recovers those answers.
    """
    text = _reasoning(raw)
    low = text.lower()
    spots = []
    for element in q.elements:
        pos = low.find(element.lower())
        if pos < 0:
            return {}
        spots.append((pos, element))
    spots.sort()
    result = {}
    for i, (pos, element) in enumerate(spots):
        end = spots[i + 1][0] if i + 1 < len(spots) else len(text)
        window = _stems(text[pos + len(element):end])
        scores = {l: len(window & _stems(t)) for l, t in q.options.items()}
        best = max(scores.values(), default=0)
        winners = [l for l, s in scores.items() if s == best]
        if best == 0 or len(winners) > 1:
            return {}
        result[element] = winners[0]
    return result


def matching(q, raw):
    allowed = q.letters
    tail = answer_tail(raw) or raw
    by_content = _match_by_content(q, raw)
    if len(by_content) == len(q.elements):
        return "; ".join(f"{e}={by_content[e]}" for e in q.elements)
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


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


def _roman(s):
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b in _ROMAN and _ROMAN[b] > v else v
    return total


def _date_value(segment):
    """First year in a text segment ('1410 r.', '44 p.n.e.'), else a century ('XV w.')."""
    for m in re.finditer(r"(\d{1,4})\s*(?:r\.|roku|rok)?\s*(p\.\s*n\.\s*e\.)?", segment):
        if len(m.group(1)) >= 3 or m.group(2):  # skip day numbers such as "1 września"
            year = int(m.group(1))
            return -year if m.group(2) else year
    m = re.search(r"\b([IVXL]+)\s*(?:w\.|wiek)(\s*p\.\s*n\.\s*e\.)?", segment)
    if m:
        mid = _roman(m.group(1)) * 100 - 50
        return -mid if m.group(2) else mid
    return None


def _order_by_dates(q, raw):
    """Sort by the dates the model assigned to each letter ('A: 1526 r., B: 966 r.')."""
    text = _reasoning(raw)
    marks = [m for m in re.finditer(rf"(?<![{_WORD}])([A-H])\s*[:)]", text) if m.group(1) in q.options]
    dates = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        value = _date_value(text[m.end():end])
        if value is not None and m.group(1) not in dates:
            dates[m.group(1)] = value
    if set(dates) != set(q.options) or len(set(dates.values())) < 2:
        return ""
    return ",".join(sorted(dates, key=lambda l: (dates[l], l)))


def order(q, raw):
    allowed = q.letters
    by_dates = _order_by_dates(q, raw)
    if by_dates:
        return by_dates
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
            value = safe_eval(m.group(1).split("=")[0])
            # "How many years passed" is never negative, whichever way round the model subtracted.
            if value < 0 and re.search(r"\bile\b|minęł|upłynęł|trwał", q.text, re.I):
                value = -value
            return format_number(value)
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
