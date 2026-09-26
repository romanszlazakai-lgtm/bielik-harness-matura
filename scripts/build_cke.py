"""Build a local knowledge base from CKE exam papers, marking schemes and other exam materials.

    python scripts/build_cke.py --src ../sources/wiedza
    python scripts/build_cke.py --src ../sources/wiedza --holdout 2023-05   # keep one exam out of RAG

Reads every file in --src:
    *zasady*.pdf           marking scheme (answers, scoring rules)          -> key
    MHIP-R0-100-YYMM.pdf   exam sheet, standard form (other forms skipped)  -> sheet
    *.rtf                  OCR export of an exam sheet (FineReader)         -> sheet
    anything else          exam materials (examiner advice, guides)         -> material

Writes to data/cke/ (git-ignored: CKE papers quote copyrighted sources, so they stay local):
    tasks.jsonl     one record per task: exam, task, type, question, sources, answer, canonical, scoring, requirement
    chunks.jsonl    RAG passages: task + answer, source texts, materials
    index.pkl       BM25 index over chunks (the held-out exam, if any, is left out)
    fewshot.json    worked examples chosen in data/cke_fewshot.json, rendered from the local papers
    heldout.jsonl   closed tasks of the held-out exam in the dev-set format, for an honest check

Source documents are public on cke.gov.pl (Egzamin maturalny > Arkusze egzaminacyjne > historia):
the May 2023 extended-level history paper (MHIP-R0-100-2305) with its marking scheme, and the
March 2022 sample paper with its marking scheme (MHIP-R0-...-2203-zasady).
"""
import argparse
import json
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from harness.retrieval import BM25  # noqa: E402
from harness.rtf import rtf_to_text  # noqa: E402
from harness.sheet import DASH, clean, parse_sheet  # noqa: E402,F401  (shared with the exam loader)

OUT = ROOT / "data" / "cke"
MONTHS = {"stycznia": 1, "lutego": 2, "marca": 3, "kwietnia": 4, "maja": 5, "czerwca": 6, "lipca": 7,
          "sierpnia": 8, "września": 9, "października": 10, "listopada": 11, "grudnia": 12}


# ---------- reading ----------

def read_text(path):
    if path.suffix.lower() == ".pdf":
        exe = shutil.which("pdftotext")
        if not exe:
            sys.exit("pdftotext (poppler) is required to read PDFs")
        out = subprocess.run([exe, "-enc", "UTF-8", str(path), "-"], capture_output=True)
        return out.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    if path.suffix.lower() == ".rtf":
        return rtf_to_text(path.read_text(encoding="latin-1"))
    return path.read_text(encoding="utf-8", errors="replace")


def exam_id(path, text):
    m = re.search(r"-(\d{2})(\d{2})(?:-zasady)?\.pdf$", path.name)
    if m:
        return f"20{m.group(1)}-{m.group(2)}"
    m = re.search(r"termin:\s*\d{1,2}\s+(\w+)\s+(\d{4})", text)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(2)}-{MONTHS[m.group(1).lower()]:02d}"
    return path.stem


def role(path):
    name = path.name.lower()
    if "zasady" in name:
        return "key"
    if path.suffix.lower() == ".rtf" or re.search(r"mhip-r0-100-\d{4}\.pdf$", name):
        return "sheet"
    if re.search(r"mhip-r0-\w{3}-\d{4}\.pdf$", name):
        return "skip"  # adapted forms of the same paper
    return "material"


# ---------- sheets ----------

# ---------- marking schemes ----------

_KEY_HEAD = re.compile(rf"Zadanie (\d+(?:\.\d+)?)\.\s*\(0{DASH}(\d)\)")
_ANSWER = re.compile(r"(?:^|\n)\s*(Rozwiązanie|Przykładowe rozwiązani[ae]|Przykładowa odpowiedź|Poprawna odpowiedź)\b:?\s*")
_GENERAL = ("Chronologia historyczna", "Analiza i interpretacja", "Tworzenie narracji")


def parse_key(text):
    parts = _KEY_HEAD.split(text)
    out = {}
    for tid, points, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        body = clean(body)
        req = ""
        for m in re.finditer(r"^([IVXL]+\.\s+[^\n]+)", body[: body.find("Zasady oceniania")], re.M):
            if not m.group(1).split(". ", 1)[1].startswith(_GENERAL):
                req = m.group(1).strip()
                break
        scoring = ""
        m = re.search(r"Zasady oceniania\s*(.+?)(?=\n(?:Rozwiązanie|Przykładow|Poprawna|Uwag)|\Z)", body, re.S)
        if m:
            scoring = m.group(1).strip()
        m = _ANSWER.search(body)
        answer = body[m.end():].strip() if m else ""
        answer = re.split(r"\n(?:Uwaga|Uwagi)\b", answer)[0].strip()
        out[tid] = {"points": int(points), "answer": answer, "scoring": scoring, "requirement": req}
    return out


def canonical(answer):
    """Map a marking-scheme answer onto the exam's auto-graded formats, where possible."""
    flat = re.sub(r"\s+", " ", answer).strip()
    if re.fullmatch(r"[A-F]", flat):
        return "single", flat, []
    tf = re.findall(rf"(\d)\s*{DASH}\s*([PF])\b", flat)
    if len(tf) >= 2:
        return "tflist", ",".join(v for _, v in tf), []
    pairs = re.findall(rf"\b([A-F])\s*{DASH}\s*(\d)\b", flat)
    if len(pairs) >= 2:
        return "matching", "; ".join(f"{a}={b}" for a, b in pairs), []
    pairs = re.findall(rf"\b(\d)\s*{DASH}\s*([A-F])\b", flat)
    if len(pairs) >= 2:
        return "matching", "; ".join(f"{a}={b}" for a, b in pairs), []
    m = re.match(r"Rozstrzygnięcie:\s*(.+?)\s+Przykładowe", flat)
    if m:
        return "decision", m.group(1).strip(" ."), []
    if "•" not in flat and len(flat.split()) <= 8 and ":" not in flat:
        bare = re.sub(r"\s+,", ",", re.sub(r"\s*\[[^\]]*\]\s*", " ", flat)).strip(" ,.")
        full = re.sub(r"[\[\]]", "", flat).strip(" ,.")
        alts = [a.strip() for m in re.findall(r"\[([^\]]*)\]", flat) for a in m.split(",")]
        # "Engels, Marks" lists alternatives: either one is a full answer.
        parts = [p.strip() for p in bare.split(",") if p.strip()]
        return "short", bare, sorted({bare, full, *parts, *[a for a in alts if len(a.split()) > 1]})
    return "open", flat, []


def question_type(question):
    low = question.lower()
    if "zaznacz p" in low or "prawdziwość" in low:
        return "tflist"
    if "przyporządkuj" in low or "dopasuj" in low:
        return "matching"
    if "uporządkuj" in low:
        return "order"
    if "właściwą odpowiedź" in low or "dokończ zdanie" in low:
        return "single"
    return None


# ---------- passages ----------

def split_words(text, n=120):
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def task_passage(exam, tid, rec):
    q = re.sub(r"\s+", " ", rec["question"])
    a = re.sub(r"\s+", " ", rec["answer"])
    # The instruction closes the task text; sources embedded before it are indexed separately.
    text = f"Polecenie: {q[-300:]} | Odpowiedź wg zasad oceniania CKE: {a[:300]}"
    if rec.get("requirement"):
        text += f" | Temat: {rec['requirement'][:120]}"
    return {"title": f"CKE {exam} zad. {tid}", "text": text, "src": "cke", "exam": exam}


# ---------- dev-format rendering ----------

def render_tflist(rec):
    """CKE true/false tables -> '(1) ... (2) ...' statements, as the exam script words them."""
    body = rec["question"]
    m = re.search(r"fałszywe\.", body)
    if not m:
        return None
    # Statements end with the answer boxes: "PF" in the PDF text, "| P | F |" in the OCR export.
    parts = re.split(r"\|?\s*\bP\s*\|?\s*F\b\s*\|?", body[m.end():])
    stmts = []
    for part in parts:
        part = re.sub(r"\n\s*\d\.\s+", " ", "\n" + part)          # stray row numbers
        part = re.sub(r"^\s*\d\.\s*\|?\s*|\|", " ", part.strip())   # leading "1. |" and cell bars
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) > 15:
            stmts.append(part)
    if len(stmts) < 2:
        return None
    lines = " ".join(f"({i}) {s}" for i, s in enumerate(stmts, 1))
    return f"Oceń prawdziwość stwierdzeń. Wpisz P (prawda) lub F (fałsz). {lines} Odpowiedz literami P/F w kolejności."


def render_single(rec):
    body = re.sub(r"\s+", " ", rec["question"])
    m = re.search(r"A\.\s+(.+?)\s+B\.\s+(.+?)\s+C\.\s+(.+?)\s+D\.\s+(.+?)(?:\s*$|\s+(?=Zadanie))", body)
    if not m:
        return None, None
    stem = body[: m.start()].strip()
    stem = re.sub(r"^(Dokończ zdanie\.|Zaznacz właściwą odpowiedź spośród podanych\.)\s*", "", stem)
    stem = re.sub(r"\s*Zaznacz właściwą odpowiedź spośród podanych\.\s*", " ", stem).strip()
    return stem, {k: v.strip(" .") for k, v in zip("ABCD", m.groups())}


def dev_item(exam, tid, rec, source_chars=1500):
    """Closed task -> dev-set item, or None when it does not fit an auto-graded format."""
    kind, key, accept = rec["canonical_type"], rec["canonical"], rec["accept"]
    sources = re.sub(r"\s+", " ", rec["sources"])[:source_chars]
    prefix = f"Źródło: {sources}\n" if sources else ""
    if kind == "single":
        stem, options = render_single(rec)
        if not options:
            return None
        return {"id": f"CKE-{exam}-{tid}", "type": "single", "question": prefix + stem, "options": options, "answer": key}
    if kind == "tflist":
        text = render_tflist(rec)
        return text and {"id": f"CKE-{exam}-{tid}", "type": "tflist", "question": prefix + text, "answer": key}
    if kind == "short":
        q = re.sub(r"\s+", " ", rec["question"]) + " Odpowiedz jak najkrócej."
        return {"id": f"CKE-{exam}-{tid}", "type": "short", "question": prefix + q, "answer": key, "accept": accept}
    return None


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=str(ROOT.parent / "sources" / "wiedza"))
    parser.add_argument("--holdout", default="", help="exam id kept out of the index, e.g. 2023-05")
    args = parser.parse_args()

    src = Path(args.src)
    OUT.mkdir(parents=True, exist_ok=True)
    sheets, keys, materials = {}, {}, []
    for path in sorted(src.iterdir()):
        if path.suffix.lower() not in (".pdf", ".rtf", ".txt", ".md") or role(path) == "skip":
            continue
        text = read_text(path)
        kind, exam = role(path), exam_id(path, text)
        if kind == "sheet":
            sheets[exam] = parse_sheet(text)
        elif kind == "key":
            keys[exam] = parse_key(text)
        else:
            materials.append((path.stem, clean(text)))
        print(f"{kind:8} {exam:10} {path.name}")

    tasks, chunks = [], []
    for exam, key in sorted(keys.items()):
        sheet = sheets.get(exam, {})
        for tid, k in key.items():
            s = sheet.get(tid, {"question": "", "sources": ""})
            kind, canon, accept = canonical(k["answer"])
            rec = {"exam": exam, "task": tid, "points": k["points"], **s, **k,
                   "canonical_type": kind, "canonical": canon, "accept": accept,
                   "question_type": question_type(s["question"])}
            tasks.append(rec)
            if exam != args.holdout:
                chunks.append(task_passage(exam, tid, rec))
        if exam != args.holdout:
            seen = set()
            for tid, s in sheet.items():
                group = tid.split(".")[0]
                # Group header sources, or the body of a standalone task that embeds its sources.
                body = s["sources"] or (s["question"] if "." not in tid and len(s["question"]) > 400 else "")
                if body and group not in seen:
                    seen.add(group)
                    for part in split_words(body):
                        chunks.append({"title": f"CKE {exam} zad. {group}: źródła", "text": part,
                                       "src": "cke", "exam": exam})
    for stem, text in materials:
        for part in split_words(text):
            chunks.append({"title": stem.replace("_", " "), "text": part, "src": "cke", "exam": "material"})

    with open(OUT / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(OUT / "index.pkl", "wb") as f:
        pickle.dump(BM25().build(chunks), f, protocol=pickle.HIGHEST_PROTOCOL)

    # Few-shot examples: ids and one-line reasoning live in the repo, task text comes from local papers.
    chosen = json.loads((ROOT / "data" / "cke_fewshot.json").read_text(encoding="utf-8"))
    by_id = {(t["exam"], t["task"]): t for t in tasks}
    shots = []
    for c in chosen:
        t = by_id.get((c["exam"], c["task"]))
        item = t and dev_item(c["exam"], c["task"], t, source_chars=800)  # keep prompts short on CPU
        if not item:
            print(f"few-shot {c['exam']} {c['task']}: not found or not renderable, skipped")
            continue
        text = item["question"]
        if item.get("options"):
            text += "\n" + "\n".join(f"{k}) {v}" for k, v in item["options"].items())
        shots.append({"id": item["id"], "type": item["type"], "question": text,
                      "answer": f"{c['reasoning']}\nOdpowiedź: {item['answer']}"})
    (OUT / "fewshot.json").write_text(json.dumps(shots, ensure_ascii=False, indent=2), encoding="utf-8")

    held = [d for t in tasks if t["exam"] == args.holdout for d in [dev_item(t["exam"], t["task"], t)] if d]
    with open(OUT / "heldout.jsonl", "w", encoding="utf-8") as f:
        for d in held:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    kinds = {}
    for t in tasks:
        kinds[t["canonical_type"]] = kinds.get(t["canonical_type"], 0) + 1
    print(f"\n{len(tasks)} tasks {kinds}; {len(chunks)} passages; {len(shots)} few-shot examples; "
          f"{len(held)} held-out items ({args.holdout or 'none'})")


if __name__ == "__main__":
    main()
