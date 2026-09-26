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
def extension(call):
    """Each extension brings new sentences, as a real model would (identical ones get deduplicated)."""
    body = " ".join(f"Fakt {call}.{i}: ustawa o miastach z 1791 r. dała mieszczanom prawo nabywania ziemi."
                    for i in range(1, 16))
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

total = len(DETECT) + 1 + 2 + 3 + 9
print(f"{total - failures}/{total} checks passed")
sys.exit(1 if failures else 0)
