"""Answer a whole exam sheet from a file, offline, and write the answers to a file.

    python main.py --questions exam.json --out answers.json            # harness
    python main.py --questions exam.json --out base.json --mode base   # untouched model
    python main.py --questions exam.json --out answers.json --resume   # continue after a crash

Input: {"questions": [...]}, a bare JSON list, or JSON Lines. Each question needs "id" and
"question"; "type" (single, multi, tflist, matching, order, numeric, short) and "options"
({"A": "..."} or a list) are used when present.

Output formats (--format):
    answers  {"answers": [{"question_id": ..., "given": ...}]}   default; field names as in the
             organisers' results table (question_id, given)
    map      {"<id>": "<answer>", ...}
    jsonl    one {"id": ..., "answer": ...} per line
    csv      id,answer

For the live exam the organisers' script queries a model endpoint instead; for that use
`python -m harness.server --port 8000`, which runs exactly the same pipeline.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from harness import config, llm, qtypes  # noqa: E402
from harness.solve import solve  # noqa: E402


def load_questions(path):
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8-sig")
    raw = raw.strip()
    if raw.startswith("{") and '"questions"' in raw[:2000]:
        return json.loads(raw)["questions"]
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def answer_base(item):
    """The untouched model: question text in, reply out, nothing else."""
    reply = llm.chat([{"role": "user", "content": qtypes.format_question(item)}], temperature=0.0)
    return {"answer": reply.strip(), "type": item.get("type"), "raw": [reply]}


def write_answers(rows, path, fmt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "answers":
        data = {"answers": [{"question_id": r["id"], "given": r["answer"]} for r in rows]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "map":
        path.write_text(json.dumps({r["id"]: r["answer"] for r in rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    elif fmt == "jsonl":
        path.write_text("".join(json.dumps({"id": r["id"], "answer": r["answer"]}, ensure_ascii=False) + "\n"
                                for r in rows), encoding="utf-8")
    elif fmt == "csv":
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "answer"])
            w.writerows([r["id"], r["answer"]] for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, help="exam file, or - for stdin")
    parser.add_argument("--out", required=True, help="where to write the answers")
    parser.add_argument("--format", choices=["answers", "map", "jsonl", "csv"], default="answers")
    parser.add_argument("--mode", choices=["harness", "base"], default="harness")
    parser.add_argument("--resume", action="store_true", help="skip questions already in the trace file")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = config.OUTPUT_DIR / f"exam_trace_{args.mode}_{Path(args.out).stem}.jsonl"

    done = {}
    if args.resume and trace_path.exists():
        for line in open(trace_path, encoding="utf-8"):
            r = json.loads(line)
            done[str(r["id"])] = r
    print(f"{len(questions)} questions, mode {args.mode}, model {config.LLM_MODEL} at {config.LLM_BASE_URL}"
          + (f", {len(done)} already answered" if done else ""))

    rows, start = [], time.time()
    with open(trace_path, "a" if args.resume else "w", encoding="utf-8") as trace:
        for n, item in enumerate(questions, 1):
            qid = str(item.get("id", n))
            if qid in done:
                rows.append(done[qid])
                continue
            if args.mode == "base":
                res = answer_base(item)
            else:
                res = solve(qtypes.format_question(item), declared_type=item.get("type"),
                            has_image=bool(item.get("images") or item.get("image")))
            row = {"id": qid, "answer": res["answer"], "type": res.get("type"),
                   "seconds": res.get("seconds"), "time": datetime.now().isoformat(timespec="seconds"),
                   "raw": res.get("raw")}
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace.flush()
            rows.append(row)
            # Write after every question, so a crash never loses finished answers.
            write_answers(rows, args.out, args.format)
            print(f"[{n}/{len(questions)}] {qid} ({row['type']}): {row['answer'][:60]!r}")

    write_answers(rows, args.out, args.format)
    print(f"done: {len(rows)} answers in {time.time() - start:.0f}s -> {args.out} ({args.format}); trace {trace_path}")


if __name__ == "__main__":
    main()
