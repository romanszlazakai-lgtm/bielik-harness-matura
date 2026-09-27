"""Answer an organisers' exam package (exam.json + images/) and write answers.json.

    python run_exam.py --exam path/to/exam.json --out answers.json               # harness
    python run_exam.py --exam path/to/exam.json --out base_answers.json --mode base
    python run_exam.py --exam path/to/exam.json --out answers.json --resume      # after a crash

Package format "separate-text-and-images-v1": items with id, group, max_points, question,
source_text, images and answer_format. Output: {"exam_id", "answers": [{"id", "answer"}]} with
every item id, answers as strings, "" when there is no answer.

The answer_format decides the pipeline:
    "A"                      single choice                  -> "C"
    "1: P\\n2: F\\n3: P"     true/false per statement       -> "1: P\\n2: F\\n3: P"
    "1: A\\n2: A"            several single choices         -> "1: B\\n2: C"
    "A: 1\\nB: 1"            fill a table with numbers      -> "A: 3\\nB: 2"
    essay (item 26)          essay pipeline                 -> "Wybrany temat nr N: ...\\n\\n<essay>"
    "Tekst po polsku..."     decision, short or open answer by the question's wording
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from harness import config, llm, prompts, qtypes, retrieval  # noqa: E402
from harness.sheet import render_single  # noqa: E402
from harness.solve import solve  # noqa: E402

_KEYED = re.compile(r"^\s*([A-Z0-9]{1,3})\s*:\s*(\S+)\s*$")
_SHORT = re.compile(r"^\s*Podaj (?:stosowaną w historiografii )?(?:nazw|nazwisk|imi)", re.I)
_NOT_SHORT = re.compile(r"\bdw(?:a|ie|óch)\b|\boraz\b|\bcech|uzasadnij|wyjaśnij", re.I)


_EMPTY_FIELDS = re.compile(r"(\n[ \t]*[A-ZĄĆĘŁŃÓŚŹŻ][^\n:]{0,30}:[ \t]*)+\s*$")


def task(item):
    """The question without the sheet's empty answer fields ("Nazwisko:\\nWyjaśnienie:"), which the
    1.5B model otherwise repeats in a loop."""
    return _EMPTY_FIELDS.sub("", item["question"].rstrip()).strip()


def item_text(item):
    """Sources first, then the task; image markers stay so the model knows a picture was there."""
    src = (item.get("source_text") or "").strip()
    return f"{src}\n\n{task(item)}" if src else task(item)


def keyed_template(fmt):
    """'1: P\\n2: F' -> [('1', 'P'), ('2', 'F')], or None when the format is free text."""
    lines = [l for l in fmt.strip().splitlines() if l.strip()]
    pairs = [_KEYED.match(l) for l in lines]
    if len(lines) >= 2 and all(pairs):
        return [(m.group(1), m.group(2)) for m in pairs]
    return None


def statements(question):
    """'1. text\\ncontinued\\n2. text' -> ['text continued', 'text'] (numbered statements)."""
    parts = re.split(r"\n\s*(?=\d+\.\s)", "\n" + question)
    out = []
    for part in parts:
        m = re.match(r"\s*(\d+)\.\s+(.+)", part, re.S)
        if m:
            out.append(re.sub(r"\s+", " ", m.group(2)).strip())
    return out


def solve_tflist(item, keys):
    q = task(item)
    head = q.split("\n1.")[0] if "\n1." in q else q
    stmts = statements(q[len(head):]) or statements(q)
    text = (f"{(item.get('source_text') or '').strip()}\n\n{head.strip()} "
            + " ".join(f"({i}) {s}" for i, s in enumerate(stmts, 1))
            + " Odpowiedz literami P/F w kolejności.").strip()
    res = solve(text, declared_type="tflist", has_image=bool(item["images"]))
    verdicts = [v.strip() for v in res["answer"].split(",") if v.strip()]
    verdicts += ["P"] * (len(keys) - len(verdicts))
    return "\n".join(f"{k}: {v}" for k, v in zip(keys, verdicts)), res


def solve_multi_single(item, keys):
    """'Dokończ zdania 1. i 2.': each numbered sentence with its own A-D options."""
    q = task(item)
    parts = [p for p in re.split(r"\n\s*(?=\d+\.\s)", q) if re.match(r"\s*\d+\.\s", p)]
    lines, traces = [], []
    for key, part in zip(keys, parts + [""] * len(keys)):
        stem = re.sub(r"^\s*\d+\.\s*", "", part)
        rendered = render_single(stem) or stem
        text = f"{(item.get('source_text') or '').strip()}\n\n{rendered}".strip()
        res = solve(text, declared_type="single", has_image=bool(item["images"]))
        lines.append(f"{key}: {res['answer'] or 'A'}")
        traces.append(res)
    return "\n".join(lines), {"answer": "\n".join(lines), "type": "multi-single", "raw": [t["raw"] for t in traces]}


def solve_keyed_values(item, template):
    """A table to fill, e.g. 'A: <fragment number>'. The model writes one 'key: value' line each."""
    keys = [k for k, _ in template]
    q = qtypes.Question(text=item_text(item), qtype="short", has_image=bool(item["images"]))
    passages = retrieval.retrieve(q)
    fmt = "\n".join(f"{k}: <{'numer' if v.isdigit() else 'odpowiedź'}>" for k, v in template)
    messages = [
        {"role": "system", "content": prompts.SYSTEM},
        # Answer lines only: asked to recall facts first, the model spent its tokens and never answered.
        {"role": "user", "content": f"{prompts._context_block(passages)}Pytanie: {item_text(item)}\n\n"
                                    f"Nie uzasadniaj. Odpowiedz wyłącznie liniami w tym formacie:\n{fmt}"},
    ]
    raw = llm.chat(messages, temperature=0.0, max_tokens=120)
    values = {}
    for line in raw.splitlines():
        m = re.match(r"\s*([A-Z0-9]{1,3})\s*[:=.)-]\s*(.+?)\s*$", line)
        if m and m.group(1) in keys and m.group(1) not in values:
            values[m.group(1)] = m.group(2)
    out = []
    for key, example in template:
        value = values.get(key, "")
        if example.isdigit():
            digits = re.findall(r"\d+", value)
            value = digits[0] if digits else "1"  # a blank scores nothing; a guess sometimes does
        out.append(f"{key}: {value}")
    return "\n".join(out), {"answer": "\n".join(out), "type": "keyed", "raw": [raw]}


def text_type(item):
    q = task(item)
    if "rozstrzygnij" in q.lower():
        return "open"                       # decision: "Rozstrzygnięcie: ... Uzasadnienie: ..."
    if _SHORT.match(q) and not _NOT_SHORT.search(q):
        return "short"
    return "open"


def answer_item(item):
    fmt = item.get("answer_format") or ""
    template = keyed_template(fmt)
    if str(item.get("group")) == "26" or "wypracowani" in fmt.lower() or "numer wybranego tematu" in fmt.lower():
        res = solve(item_text(item), declared_type="essay")
        return res["answer"], res
    if template and all(v in ("P", "F") for _, v in template):
        return solve_tflist(item, [k for k, _ in template])
    if template and all(re.fullmatch(r"[A-H]", v) for _, v in template) and all(k.isdigit() for k, _ in template):
        return solve_multi_single(item, [k for k, _ in template])
    if template:
        return solve_keyed_values(item, template)
    if re.fullmatch(r"\s*[A-H]\s*", fmt):
        rendered = render_single(task(item)) or task(item)
        src = (item.get("source_text") or "").strip()
        res = solve(f"{src}\n\n{rendered}".strip(), declared_type="single", has_image=bool(item["images"]))
        return res["answer"], res
    kind = text_type(item)
    # Open answers worth more points may keep more sentences (one per point, plus one).
    config.OPEN_MAX_SENTENCES = max(2, int(item.get("max_points") or 1) + 1)
    res = solve(item_text(item), declared_type=kind, has_image=bool(item["images"]))
    return res["answer"], res


def answer_base(item, instructions):
    """The untouched model: exam instructions, sources, task and answer format; reply as given."""
    prompt = (f"{instructions}\n\n{item_text(item)}\n\nFormat odpowiedzi: {item.get('answer_format', '')}").strip()
    tokens = config.ESSAY_MAX_TOKENS if str(item.get("group")) == "26" else 400
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=tokens,
                   timeout=config.ESSAY_TIMEOUT)
    return raw.strip(), {"answer": raw.strip(), "type": "base", "raw": [raw]}


def write_answers(path, exam_id, ids, answers):
    data = {"exam_id": exam_id, "answers": [{"id": i, "answer": str(answers.get(i, ""))} for i in ids]}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exam", required=True, help="exam.json of the package")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["harness", "base"], default="harness")
    parser.add_argument("--only", default="", help="comma-separated item ids, for a quick test")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    exam = json.loads(Path(args.exam).read_text(encoding="utf-8"))
    template_path = Path(args.exam).parent / "answers-template.json"
    ids = ([a["id"] for a in json.loads(template_path.read_text(encoding="utf-8"))["answers"]]
           if template_path.exists() else [it["id"] for it in exam["items"]])
    items = [it for it in exam["items"] if not args.only or it["id"] in args.only.split(",")]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = config.OUTPUT_DIR / f"exam_{exam['exam_id']}_{args.mode}_trace.jsonl"
    answers = {}
    if args.resume and Path(args.out).exists():
        answers = {a["id"]: a["answer"] for a in json.loads(Path(args.out).read_text(encoding="utf-8"))["answers"]
                   if a["answer"]}
    print(f"{exam['exam_id']}: {len(items)} items, mode {args.mode}, model {config.LLM_MODEL} at {config.LLM_BASE_URL}")

    start = time.time()
    with open(trace_path, "a", encoding="utf-8") as trace:
        for n, item in enumerate(items, 1):
            if item["id"] in answers:
                continue
            t0 = time.time()
            try:
                answer, res = (answer_base(item, exam.get("instructions", "")) if args.mode == "base"
                               else answer_item(item))
            except Exception as exc:  # one failing item must not cost the others
                answer, res = "", {"error": repr(exc)}
            answers[item["id"]] = answer
            trace.write(json.dumps({"id": item["id"], "answer": answer, "seconds": round(time.time() - t0, 1),
                                    "type": res.get("type"), "raw": res.get("raw"), "error": res.get("error")},
                                   ensure_ascii=False) + "\n")
            trace.flush()
            write_answers(args.out, exam["exam_id"], ids, answers)  # after every item: a crash loses nothing
            shown = answer.replace("\n", " | ")[:70]
            print(f"[{n}/{len(items)}] {item['id']:5} {time.time() - t0:5.1f}s  {shown!r}")

    write_answers(args.out, exam["exam_id"], ids, answers)
    blanks = [i for i in ids if not answers.get(i)]
    print(f"done in {time.time() - start:.0f}s -> {args.out}; {len(ids) - len(blanks)}/{len(ids)} answered"
          + (f"; blank: {blanks}" if blanks else ""))


if __name__ == "__main__":
    main()
