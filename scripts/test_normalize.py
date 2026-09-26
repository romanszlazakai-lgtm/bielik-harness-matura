"""Model-free checks of question parsing and answer normalization. Run: python scripts/test_normalize.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from harness import normalize, qtypes  # noqa: E402
from harness.solve import combine  # noqa: E402

ORDER_Q = ("Uporządkuj ery geologiczne od najstarszej do najmłodszej. A) mezozoik; B) kenozoik; "
           "C) paleozoik; D) proterozoik. Podaj litery w kolejności, np. C,A,D,B.")
MATCH_Q = ("Przyporządkuj skałom ich typ genetyczny. Elementy: granit, bazalt, marmur, wapień. Kategorie: "
           "A) magmowa głębinowa; B) magmowa wylewna; C) metamorficzna; D) osadowa organogeniczna. "
           "Odpowiedz w formacie element=litera, np. granit=?; bazalt=?; marmur=?; wapień=?.")
TF_Q = ("Oceń prawdziwość stwierdzeń. Wpisz P (prawda) lub F (fałsz). (1) Pasaty wieją od wyżów. "
        "(2) Prąd Humboldta jest ciepły. (3) Monsun wieje znad oceanu. Odpowiedz literami P/F w kolejności, np. P,F,P.")
SINGLE_Q = "Która ustawa zniosła liberum veto?\nA) Sejm Niemy\nB) Prawa kardynalne\nC) Sejm Wielki\nD) Uniwersał połaniecki"
MULTI_Q = ("Zaznacz wszystkie państwa, w których walutą jest euro. (Może być kilka poprawnych, podaj wszystkie litery.)"
           "\nA) Polska\nB) Niemcy\nC) Czechy\nD) Słowacja")
NUM_Q = "Źródło rzeki leży na 800 m, ujście na 200 m, długość 120 km. Oblicz średni spadek w promilach. Podaj samą liczbę (w ‰)."
SHORT_Q = "Podaj nazwę procesu rozpuszczania wapieni przez wodę. Odpowiedz jak najkrócej."

CASES = [
    # (question, raw model output, expected normalized answer)
    (SINGLE_Q, "Liberum veto zniósł Sejm Wielki w 1791 r.\nOdpowiedź: C", "C"),
    (SINGLE_Q, "C) Sejm Wielki", "C"),
    (SINGLE_Q, "Prawidłowa odpowiedź to Sejm Wielki.", "C"),
    (SINGLE_Q, "Odpowiedź: **C**", "C"),
    (MULTI_Q, "Euro jest w Niemczech i na Słowacji.\nOdpowiedź: D, B", "B,D"),
    (TF_Q, "(1) prawda (2) fałsz (3) prawda\nOdpowiedź: P,F,P", "P,F,P"),
    (TF_Q, "1. Prawda\n2. Fałsz\n3. Prawda", "P,F,P"),
    (MATCH_Q, "Odpowiedź: granit=A; bazalt=B; marmur=C; wapień=D", "granit=A; bazalt=B; marmur=C; wapień=D"),
    (MATCH_Q, "Granit - A, bazalt - B, marmur - C, wapień - D", "granit=A; bazalt=B; marmur=C; wapień=D"),
    (MATCH_Q, "Odpowiedź: granit=A; bazalt=B; marmur=C", "granit=A; bazalt=B; marmur=C; wapień=D"),
    (ORDER_Q, "D: proterozoik, C: paleozoik, A: mezozoik, B: kenozoik\nOdpowiedź: D,C,A,B", "D,C,A,B"),
    (ORDER_Q, "Odpowiedź: D C A", "D,C,A,B"),
    (NUM_Q, "WYRAŻENIE: (800-200)/120\nOdpowiedź: 5", "5"),
    (NUM_Q, "Spadek wynosi 600 m / 120 km = 5‰.\nOdpowiedź: 5 ‰", "5"),
    (SHORT_Q, "To proces krasowienia.\nOdpowiedź: krasowienie.", "krasowienie"),
]

TYPE_CASES = [(ORDER_Q, "order"), (MATCH_Q, "matching"), (TF_Q, "tflist"), (SINGLE_Q, "single"),
              (MULTI_Q, "multi"), (NUM_Q, "numeric"), (SHORT_Q, "short")]


def main():
    failures = 0
    for text, expected in TYPE_CASES:
        got = qtypes.parse(text).qtype
        if got != expected:
            failures += 1
            print(f"FAIL type: expected {expected}, got {got}: {text[:60]!r}")
    q = qtypes.parse(ORDER_Q)
    if sorted(q.options) != list("ABCD") or q.options["D"] != "proterozoik":
        failures += 1
        print(f"FAIL options: {q.options}")
    q = qtypes.parse(MATCH_Q)
    if q.elements != ["granit", "bazalt", "marmur", "wapień"] or sorted(q.options) != list("ABCD"):
        failures += 1
        print(f"FAIL matching parse: {q.elements} {q.options}")
    if len(qtypes.parse(TF_Q).statements) != 3:
        failures += 1
        print("FAIL tflist statements")

    for text, raw, expected in CASES:
        q = qtypes.parse(text)
        got = normalize.normalize(q, raw)
        if got != expected:
            failures += 1
            print(f"FAIL {q.qtype}: raw={raw!r}\n     expected {expected!r}, got {got!r}")

    votes = [("tflist", ["P,F,P", "P,P,P", "F,F,P"], "P,F,P"), ("multi", ["B,D", "B", "B,D"], "B,D"),
             ("single", ["C", "A", "A"], "A"), ("matching", ["x=A; y=B", "x=B; y=B", "x=A; y=C"], "x=A; y=B")]
    for qtype, answers, expected in votes:
        got = combine(qtype, answers)
        if got != expected:
            failures += 1
            print(f"FAIL vote {qtype}: {answers} -> {got!r}, expected {expected!r}")

    total = len(TYPE_CASES) + 3 + len(CASES) + len(votes)
    print(f"{total - failures}/{total} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
