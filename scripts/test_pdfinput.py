"""Model-free and OCR-free checks of the exam loader. Run: python scripts/test_pdfinput.py

A three-page "PDF" is simulated: page 2 has no text layer (a scan). With OCR available the loader
must OCR exactly that page; without it, it must keep the other pages and report the missing one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from harness import pdfinput, qtypes, sheet  # noqa: E402

failures = 0


def check(name, condition, detail=""):
    global failures
    if not condition:
        failures += 1
        print(f"FAIL {name} {detail}")


PAGE1 = ("Zadanie 1.\nŹródło. Fragment kroniki opisujący zjazd w Gnieźnie w roku 1000 z udziałem cesarza.\n"
         "Zadanie 1.1. (0–1)\n1.1.\n0–1\nDokończ zdanie. Zaznacz właściwą odpowiedź spośród podanych.\n"
         "Opisany zjazd odbył się za panowania A. Mieszka I. B. Bolesława Chrobrego. C. Kazimierza Odnowiciela. "
         "D. Władysława Hermana.\n")
PAGE2_OCR = ("Zadanie 2. (0–2)\nOceń prawdziwość poniższych stwierdzeń. Zaznacz P, jeśli stwierdzenie jest "
             "prawdziwe, albo F – jeśli jest fałszywe.\n1. Unia lubelska została zawarta w 1569 roku. PF\n"
             "2. Hołd pruski złożono w Malborku w 1525 roku. PF\nStrona 2 z 3\n")
PAGE3 = ("Zadanie 3. (0–1)\nWyjaśnij, dlaczego Konstytucja 3 maja zniosła liberum veto.\n"
         "..............................................................\n")

# ---- sheet parsing and normalisation (no PDF involved) ----
qs = sheet.questions_from_text(PAGE1 + PAGE2_OCR + PAGE3)
ids = [q["id"] for q in qs]
check("task ids", ids == ["Z1.1", "Z2", "Z3"], str(ids))
by_id = {q["id"]: qtypes.parse(q["question"]) for q in qs}
check("single + options", by_id["Z1.1"].qtype == "single" and by_id["Z1.1"].options.get("B") == "Bolesława Chrobrego",
      str(by_id["Z1.1"].options))
check("sources attached", "zjazd w Gnieźnie" in qs[0]["question"])
check("tflist rendered", by_id["Z2"].qtype == "tflist" and len(by_id["Z2"].statements) == 2,
      str(by_id["Z2"].statements))
check("open detected", by_id["Z3"].qtype == "open")
check("noise removed", "Strona 2 z 3" not in qs[1]["question"] and "....." not in qs[2]["question"])
check("fallback paragraphs", len(sheet.questions_from_text("1. Podaj nazwę dokumentu z 1505 r. jak najkrócej.\n"
                                                          "2. Wyjaśnij znaczenie unii w Krewie dla Litwy.")) == 2)

# ---- page routing: text layer vs OCR, with the PDF tools stubbed ----
pdfinput.text_layer_pages = lambda path: [PAGE1, "", PAGE3]      # page 2 is a scan
pdfinput.page_count = lambda path: 3
ocr_calls = []


def fake_ocr(path, page_no, out_dir):
    ocr_calls.append(page_no)
    return PAGE2_OCR


pdfinput.ocr_page = fake_ocr
text, report = pdfinput.read_pdf("exam.pdf")
check("only the scan is OCR'd", ocr_calls == [2] and report["ocr"] == 1 and report["text_layer"] == 2, str(report))
check("OCR text used", "Unia lubelska" in text)


def no_ocr(path, page_no, out_dir):
    raise RuntimeError("no OCR engine")


pdfinput.ocr_page = no_ocr
text, report = pdfinput.read_pdf("exam.pdf")
check("missing OCR reported", report["failed"] == [2] and "no OCR engine" in report["ocr_error"], str(report))
check("other pages kept", "Zadanie 1.1" in text and "Zadanie 3." in text)

pdfinput.ocr_page = fake_ocr
ocr_calls.clear()
text, report = pdfinput.read_pdf("exam.pdf", force_ocr=True)
check("force OCR", ocr_calls == [1, 2, 3], str(ocr_calls))

total = 7 + 5
print(f"{total - failures}/{total} checks passed")
sys.exit(1 if failures else 0)
