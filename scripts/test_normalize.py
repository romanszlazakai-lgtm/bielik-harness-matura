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

HIST_ORDER_Q = ("Uporządkuj wydarzenia chronologicznie od najwcześniejszego. A) hołd pruski; B) chrzest Litwy; "
                "C) potop szwedzki; D) uchwalenie Konstytucji 3 maja. Podaj litery w kolejności, np. C,A,D,B.")
WAR_ORDER_Q = ("Uporządkuj wydarzenia chronologicznie od najwcześniejszego. A) wybuch II wojny światowej; "
               "B) podpisanie traktatu wersalskiego; C) przewrót majowy; D) plebiscyt na Górnym Śląsku. "
               "Podaj litery w kolejności, np. C,A,D,B.")
STYLE_ORDER_Q = ("Uporządkuj style w sztuce od najwcześniejszego. A) barok; B) styl romański; C) renesans; "
                 "D) gotyk. Podaj litery w kolejności, np. C,A,D,B.")
DATE_MATCH_Q = ("Przyporządkuj wydarzeniom daty. Elementy: chrzest Polski, bitwa pod Grunwaldem, unia lubelska, "
                "bitwa pod Wiedniem. Kategorie: A) 1683; B) 966; C) 1569; D) 1410. Odpowiedz w formacie "
                "element=litera, np. chrzest Polski=?; bitwa pod Grunwaldem=?; unia lubelska=?; bitwa pod Wiedniem=?.")
JAG_MULTI_Q = ("Zaznacz wszystkich władców z dynastii Jagiellonów. (Może być kilka poprawnych, podaj wszystkie litery.)"
               "\nA) Kazimierz Jagiellończyk\nB) Stefan Batory\nC) Zygmunt August\nD) Władysław Warneńczyk\nE) Henryk Walezy")
YEARS_Q = ("Oblicz, ile lat minęło od uchwalenia Konstytucji 3 maja (1791) do odzyskania niepodległości "
           "przez Polskę (1918). Podaj samą liczbę.")

# Real replies from the first dev run, where the model knew the facts but assembled them wrongly.
REAL_CASES = [
    (HIST_ORDER_Q, "A: 1526 r., B: 966 r., C: 1655 r., D: 1791 r.\nOdpowiedź: A,B,C,D", "B,A,C,D"),
    (WAR_ORDER_Q, "A: 1 września 1939 r., B: 28 czerwca 1919 r., C: 15 maja 1926 r., D: 20 marca 1921 r.\n"
                  "Odpowiedź: C,A,D,B", "B,D,C,A"),
    (STYLE_ORDER_Q, "A: XVII w.\nB: XI w.\nC: XV w.\nD: XIII w.\nOdpowiedź: A,B,C,D", "B,D,C,A"),
    (DATE_MATCH_Q, "Chrzest Polski miał miejsce w 966 r., bitwa pod Grunwaldem odbyła się 15 lipca 1410 r., "
                   "unia lubelska została zawarta w 1569 r., a bitwa pod Wiedniem miała miejsce 12 września 1683 r.\n"
                   "Odpowiedź: chrzest Polski=B; bitwa pod Grunwaldem=A; unia lubelska=C; bitwa pod Wiedniem=D",
     "chrzest Polski=B; bitwa pod Grunwaldem=D; unia lubelska=C; bitwa pod Wiedniem=A"),
    (JAG_MULTI_Q, "A: TAK (Jagiellon)\nB: NIE (Batory)\nC: TAK\nD: TAK\nE: NIE\nOdpowiedź: A,B,C,D", "A,C,D"),
    (YEARS_Q, "WYRAŻENIE: 1791-1918\nOdpowiedź: 207", "127"),
]

UPRISING_MULTI_Q = ("Zaznacz wszystkie powstania, które wybuchły w XIX wieku. (Może być kilka poprawnych, podaj "
                    "wszystkie litery.)\nA) insurekcja kościuszkowska\nB) powstanie listopadowe\nC) powstanie styczniowe"
                    "\nD) powstanie wielkopolskie 1918-1919\nE) powstanie krakowskie")
YEAR_MULTI_Q = ("Zaznacz wszystkie wydarzenia, które miały miejsce w Polsce w 1989 r. (Może być kilka poprawnych, "
                "podaj wszystkie litery.)\nA) obrady Okrągłego Stołu\nB) wybory parlamentarne 4 czerwca\n"
                "C) wprowadzenie stanu wojennego\nD) powołanie rządu Tadeusza Mazowieckiego\nE) podpisanie porozumień sierpniowych")
GNIEZNO_TF_Q = ("Oceń prawdziwość stwierdzeń. Wpisz P (prawda) lub F (fałsz). (1) Chrzest Mieszka I miał miejsce w 966 r. "
                "(2) Zjazd gnieźnieński odbył się w 1000 r. (3) Pierwszą koronacją królewską w Polsce była koronacja "
                "Mieszka II. Odpowiedz literami P/F w kolejności, np. P,F,P.")

# v3 line formats and v2 replies the line parser now reads correctly.
V3_CASES = [
    (GNIEZNO_TF_Q, "1. Chrzest w 966 r. => P\n2. Zjazd gnieźnieński, 1000 r. => P\n3. Pierwszy był Bolesław Chrobry "
                   "w 1025 r. => F\nOdpowiedź: P,F,P", "P,P,F"),
    (UPRISING_MULTI_Q, "A: insurekcja kościuszkowska, 1794 r. => TAK\nB: powstanie listopadowe, 1830 r. => TAK\n"
                       "C: powstanie styczniowe, 1863 r. => TAK\nD: powstanie wielkopolskie, 1918 r. => TAK\n"
                       "E: powstanie krakowskie, 1846 r. => NIE\nOdpowiedź: A,B,C,D", "B,C,E"),
    (YEAR_MULTI_Q, "A: Okrągły Stół, luty-kwiecień 1989 r. => TAK\nB: wybory 4 czerwca 1989 r. => TAK\n"
                   "C: stan wojenny, 13 grudnia 1981 r. => NIE\nD: rząd Mazowieckiego, wrzesień 1989 r. => TAK\n"
                   "E: porozumienia sierpniowe, 1980 r. => TAK\nOdpowiedź: A,B,D,E", "A,B,D"),
    (JAG_MULTI_Q, "A: Kazimierz Jagiellończyk, syn Jagiełły => TAK\nB: Stefan Batory, książę siedmiogrodzki => NIE\n"
                  "C: Zygmunt August, ostatni Jagiellon => TAK\nD: Władysław Warneńczyk, syn Jagiełły => TAK\n"
                  "E: Henryk Walezy, z dynastii Walezjuszy => NIE\nOdpowiedź: A,C,D", "A,C,D"),
    (DATE_MATCH_Q, "Chrzest Polski: 966\nBitwa pod Grunwaldem: 1410\nUnia lubelska: 1569\nBitwy pod Wiedniem: 1683\n"
                   "Odpowiedź: chrzest Polski=D; bitwa pod Grunwaldem=C; unia lubelska=C; bitwa pod Wiedniem=A",
     "chrzest Polski=B; bitwa pod Grunwaldem=D; unia lubelska=C; bitwa pod Wiedniem=A"),
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

    from harness.prompts import _without_format_examples
    for text in (ORDER_Q, TF_Q):
        cleaned = _without_format_examples(text)
        if "C,A,D,B" in cleaned or "P,F,P" in cleaned:
            failures += 1
            print(f"FAIL format example kept: {cleaned[-80:]!r}")

    for stem, expected in (("które wybuchły w XIX wieku", (1801, 1900)), ("w Polsce w 1989 r.", (1989, 1989)),
                           ("z dynastii Jagiellonów", None), ("w V w. p.n.e.", None)):
        got = normalize._time_criterion(stem)
        if got != expected:
            failures += 1
            print(f"FAIL time criterion {stem!r}: {got}")

    for text, raw, expected in CASES + REAL_CASES + V3_CASES:
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

    total = len(TYPE_CASES) + 3 + 2 + 4 + len(CASES) + len(REAL_CASES) + len(V3_CASES) + len(votes)
    print(f"{total - failures}/{total} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
