"""Model-free checks of open-answer and essay handling. Run: python scripts/test_essay.py

The model is replaced by a stub that returns a short, cut-off draft first, so the length check,
the extension request and the clean-up are exercised exactly as they would be on the exam.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from harness import config, essay, llm, prompts, qtypes  # noqa: E402

failures = 0


def check(name, condition, detail=""):
    global failures
    if not condition:
        failures += 1
        print(f"FAIL {name} {detail}")


# ---- type detection ----
THREE_TOPICS = ("Wybierz jeden z trzech tematów i napisz wypracowanie.\nTemat 1. Oceń reformy Sejmu Wielkiego.\n"
                "Temat 2. Porównaj przyczyny powstania listopadowego i styczniowego.\nTemat 3. Scharakteryzuj reformację.")
LABELLED_TOPICS = ("Napisz wypracowanie na jeden z tematów.\nA) Znaczenie unii lubelskiej\n"
                   "B) Skutki potopu szwedzkiego\nC) Reformy Stanisława Augusta")
DETECT = [
    (THREE_TOPICS, "essay"),
    (LABELLED_TOPICS, "essay"),
    ("Przedstaw w formie rozprawki skutki rozbiorów Polski.", "essay"),
    ("Rozstrzygnij, czy unia w Krewie była unią realną. Odpowiedź uzasadnij.", "open"),
    ("Wyjaśnij, na czym polegało wypaczenie zasady liberum veto.", "open"),
    ("Podaj nazwę i wyjaśnij znaczenie dokumentu z 1505 r.", "open"),
    ("Podaj nazwę procesu. Odpowiedz jak najkrócej.", "short"),
    ("Podaj imię władcy, za panowania którego zerwano unię.", "short"),
]
for text, expected in DETECT:
    got = qtypes.parse(text).qtype
    check("detect", got == expected, f"{expected} != {got}: {text[:50]!r}")
check("alias", qtypes.parse("Temat X", declared_type="wypracowanie").qtype == "essay")

q = qtypes.parse(THREE_TOPICS)
check("topics numbered", len(q.topics) == 3 and q.topics[1].startswith("Porównaj"), str(q.topics))
q = qtypes.parse(LABELLED_TOPICS)
check("topics labelled", len(q.topics) == 3 and not q.options, str(q.topics))

# ---- prompts ----
q = qtypes.parse("Rozstrzygnij, czy unia w Krewie była unią realną. Odpowiedź uzasadnij.")
msgs = prompts.build_messages(q, [])
check("decision prompt", "Rozstrzygnięcie:" in msgs[-1]["content"] and prompts.max_tokens(q) == config.OPEN_MAX_TOKENS)
check("essay tokens", prompts.max_tokens(qtypes.parse(THREE_TOPICS)) == config.ESSAY_MAX_TOKENS)
check("essay prompt", "Argument 3:" in prompts.essay_messages("Temat", [])[-1]["content"])

# ---- essay pipeline with a stub model ----
SENT = "Sejm Wielki w latach 1788-1792 przeprowadził reformy skarbowe i wojskowe, które wzmocniły państwo."
DRAFT = ("Wstęp: Reformy Sejmu Wielkiego były próbą ratowania państwa w latach 1788-1792.\n\n"
         "Argument 1: " + " ".join([SENT] * 3) + "\n\n"        # repeated sentence must be removed
         "Argument 2: Konstytucja 3 maja 1791 r. zniosła liberum veto i wolną elekcję, co wzmacniało władzę. "
         "Argument 3: Konfederacja targowicka i wojna z Rosją w 1792 r. przerwały reformy, a potem nastąpił")  # cut off
FACTS = ["Ustawa o miastach z kwietnia 1791 r. dała mieszczanom prawo nabywania dóbr ziemskich",
         "Sejm uchwalił w 1788 r. powiększenie armii do stu tysięcy żołnierzy",
         "Ofiara dziesiątego grosza z 1789 r. była pierwszym stałym podatkiem dochodowym szlachty",
         "Rada Nieustająca została zniesiona w styczniu 1789 r. pod naciskiem stronnictwa patriotycznego",
         "Konstytucja 3 maja wprowadziła trójpodział władzy według myśli Monteskiusza",
         "Zasada dziedziczności tronu miała zakończyć zagraniczne ingerencje przy elekcjach",
         "Straż Praw skupiała władzę wykonawczą przy królu i ministrach odpowiedzialnych przed sejmem",
         "Konfederacja targowicka zawiązana w 1792 r. wezwała na pomoc armię rosyjską",
         "Wojna w obronie Konstytucji zakończyła się przystąpieniem króla do Targowicy",
         "Drugi rozbiór w 1793 r. odebrał Rzeczypospolitej ogromne obszary na wschodzie i zachodzie",
         "Sejm grodzieński w 1793 r. unieważnił postanowienia Sejmu Czteroletniego",
         "Insurekcja kościuszkowska w 1794 r. była próbą obrony reform zbrojnym powstaniem",
         "Hugo Kołłątaj i Ignacy Potocki należeli do głównych twórców ustawy rządowej",
         "Prawo o sejmikach odebrało głos szlachcie gołocie nieposiadającej ziemi",
         "Chłopi zostali wzięci pod opiekę prawa i rządu krajowego",
         "Przymierze z Prusami zawarte w 1790 r. okazało się złudną gwarancją bezpieczeństwa",
         "Katarzyna II uznała zmiany ustrojowe za zagrożenie dla wpływów rosyjskich",
         "Uniwersał połaniecki ograniczył pańszczyznę i zapewnił chłopom wolność osobistą"]


def extension(call):
    """Each extension brings new sentences, as a real model would (repeats get removed)."""
    body = " ".join(f"{fact}, co pokazuje etap {call}." for fact in FACTS)
    return f"Argument {call + 2}: {body}\n\nPodsumowanie: Reformy były szansą, ale zabrakło czasu i sił na ich obronę."


calls = []


def stub(messages, **kwargs):
    calls.append((messages, kwargs))
    return DRAFT if len(calls) == 1 else extension(len(calls))


llm.chat = stub
config.ESSAY_MIN_WORDS = 300
result = essay.solve_essay(qtypes.parse(THREE_TOPICS))
text = result["answer"]
check("extension requested", len(calls) >= 2 and "Argument 4" in calls[1][0][-1]["content"], str(len(calls)))
check("extensions capped", len(calls) <= 1 + config.ESSAY_MAX_EXTENSIONS, str(len(calls)))
check("essay timeout", calls[0][1].get("timeout") == config.ESSAY_TIMEOUT)
check("min words", result["words"] >= 300, str(result["words"]))
check("labels removed", "Argument 1:" not in text and "Wstęp:" not in text)
check("cut-off trimmed", "a potem nastąpił" not in text)
check("duplicates removed", text.count(SENT) == 1, str(text.count(SENT)))
check("topic stated", text.startswith("Wybrany temat:"))
check("has conclusion", "zabrakło czasu" in text)

# ---- clean-up of what the model actually produced in the first live test ----
MARKDOWN = ("**Temat:** **Rewolucja amerykańska i francuska miały podobne przyczyny**\n\n"
            "Reforma ustroju politycznego**\n"
            "- **Czynniki polityczne:**\n"
            "  - **Konflikt z władzą brytyjską:** Koloniści walczyli o niepodległość od Wielkiej Brytanii.\n"
            "  - **Brak reprezentacji:** Koloniści nie mieli posłów w parlamencie w Londynie\n\n"
            "- **Czynniki kulturowe:**\n"
            "  - **Idee oświecenia:** Koloniści cenili sobie wolność i równość głoszone przez oświecenie.\n\n"
            "**Wnioski:**\n"
            "- Koloniści cenlili sobie wolność i równość głoszone przez oświecenie.")
clean = essay.finalize(MARKDOWN)
check("no markdown left", "**" not in clean and "- " not in clean and "Temat:" not in clean, clean)
check("headings dropped", "Reforma ustroju politycznego" not in clean and "Czynniki polityczne" not in clean, clean)
check("bullets kept as sentences", "Koloniści walczyli o niepodległość od Wielkiej Brytanii." in clean
      and "nie mieli posłów w parlamencie w Londynie." in clean, clean)
check("near-duplicate removed", clean.count("wolność i równość") == 1, clean)

from harness import normalize  # noqa: E402
decision = qtypes.parse("Rozstrzygnij, czy hołd pruski oznaczał włączenie Prus do Korony. Odpowiedź uzasadnij.")
out = normalize.normalize(decision, "Uzasadnienie: Prusy Książęce zostały lennem Polski, a nie częścią Korony.\n"
                                    "Rozstrzygnięcie: Nie")
check("decision reordered", out.startswith("Rozstrzygnięcie: Nie\nUzasadnienie: Prusy Książęce"), out)
check("decision facts first in prompt", prompts.DECISION_FORMAT.index("Uzasadnienie") < prompts.DECISION_FORMAT.index("Rozstrzygnięcie"))

# ---- labels keep their content; extensions go before the conclusion (second live test) ----
labelled = essay.finalize("**Fakt historyczny:** **Deklaracja niepodległości Stanów Zjednoczonych (1776)**\n"
                          "**Wniosek:** Rewolucja amerykańska miała podobne przyczyny jak francuska.")
check("label content kept", "Deklaracja niepodległości Stanów Zjednoczonych (1776)." in labelled
      and "Rewolucja amerykańska miała podobne przyczyny" in labelled and "Wniosek" not in labelled, labelled)

SHORT_DRAFT = ("Wstęp: Sejm Wielki próbował ratować państwo.\n\nArgument 1: Konstytucja 3 maja zniosła liberum veto.\n\n"
               "Podsumowanie: Reformy przyszły za późno, by ocalić Rzeczpospolitą.")
calls.clear()


def stub_before_conclusion(messages, **kwargs):
    calls.append(messages)
    if len(calls) == 1:
        return SHORT_DRAFT
    return "Argument 2: " + " ".join(f"{fact}." for fact in FACTS) + "\n\nPodsumowanie: Drugie podsumowanie."


llm.chat = stub_before_conclusion
res = essay.solve_essay(qtypes.parse("Napisz wypracowanie: Oceń reformy Sejmu Wielkiego."))
text = res["answer"]
check("argument before conclusion", text.rstrip().endswith("by ocalić Rzeczpospolitą."), text[-120:])
check("no second conclusion", "Drugie podsumowanie" not in text)
check("asked for argument only", "Nie pisz podsumowania" in calls[1][-1]["content"])

total = len(DETECT) + 1 + 2 + 3 + 9 + 6 + 4
print(f"{total - failures}/{total} checks passed")
sys.exit(1 if failures else 0)
