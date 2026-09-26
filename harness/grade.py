"""Local grader for the dev set. Mirrors an automatic exam grader: one answer, right or wrong."""
import re


def _clean(s):
    return re.sub(r"[\s.;:\"'„”*_()]", "", str(s)).upper()


def _letters(s):
    return sorted(set(re.findall(r"[A-H]", str(s).upper())))


def _words(s):
    return re.sub(r"[^\w\s-]", "", str(s).lower()).split()


def is_correct(item, given):
    """True/False for auto-gradable types; None for open answers and essays (need a human or a judge)."""
    qtype, key = item["type"], item.get("answer", "")
    if qtype in ("open", "essay"):
        return None
    if not given:
        return False
    if qtype == "single":
        return _clean(given) == _clean(key)
    if qtype == "multi":
        return _letters(given) == _letters(key)
    if qtype in ("tflist", "order"):
        return _clean(given).replace(",", "") == _clean(key).replace(",", "")
    if qtype == "matching":
        def pairs(s):
            return {k.strip().lower(): v.strip().upper()
                    for k, v in (p.split("=", 1) for p in str(s).split(";") if "=" in p)}
        return pairs(given) == pairs(key)
    if qtype == "numeric":
        try:
            return abs(float(str(given).replace(",", ".")) - float(str(key).replace(",", "."))) < 0.01
        except ValueError:
            return False
    if qtype == "short":
        g = " ".join(_words(given))
        accepted = [" ".join(_words(a)) for a in item.get("accept", [key])]
        return g in accepted or (len(g.split()) <= 4 and any(a in g for a in accepted))
    return False
