"""Run the dev set through the bare model or through the harness and print the score.

    python scripts/run_dev.py --mode base       # untouched model, question text only
    python scripts/run_dev.py --mode harness    # full pipeline
    python scripts/run_dev.py --mode harness --limit 5 --votes 1   # quick check
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from harness import config, llm, normalize, qtypes  # noqa: E402
from harness.grade import is_correct  # noqa: E402
from harness.solve import solve  # noqa: E402


question_text = qtypes.format_question


def run_base(item):
    raw = llm.chat([{"role": "user", "content": question_text(item)}], temperature=0.0)
    q = qtypes.parse(question_text(item), declared_type=item["type"])
    return {"answer": raw.strip(), "lenient": normalize.normalize(q, raw), "raw": [raw]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "harness"], default="harness")
    parser.add_argument("--data", default=str(ROOT / "data" / "dev" / "historia_dev.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--votes", type=int, default=0)
    parser.add_argument("--no-type", action="store_true", help="make the harness detect the type itself")
    args = parser.parse_args()

    items = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[: args.limit]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / f"dev_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    by_type = defaultdict(lambda: [0, 0])
    score = lenient_score = 0
    ungraded = []
    start = time.time()

    with open(out_path, "w", encoding="utf-8") as out:
        for item in items:
            if args.mode == "base":
                res = run_base(item)
                ok = is_correct(item, res["answer"])
                lenient_score += bool(is_correct(item, res["lenient"]))
            else:
                res = solve(question_text(item), declared_type=None if args.no_type else item["type"],
                            votes=args.votes or None)
                ok = is_correct(item, res["answer"])
            out.write(json.dumps({**item, **res, "correct": ok}, ensure_ascii=False) + "\n")
            if ok is None:  # open answers and essays: saved for human review, not scored
                ungraded.append(item["id"])
                words = len(res["answer"].split())
                print(f"?? {item['id']} [{item['type']:8}] {words} words, {res.get('seconds', 0)}s: "
                      f"{res['answer'][:90]!r}")
                continue
            score += ok
            by_type[item["type"]][0] += ok
            by_type[item["type"]][1] += 1
            mark = "OK " if ok else "-- "
            print(f"{mark}{item['id']} [{item['type']:8}] key={item['answer']!r:28} got={res['answer'][:40]!r}")

    n = len(items) - len(ungraded)
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    for t, (ok, total) in sorted(by_type.items()):
        print(f"  {t:9} {ok:3}/{total:<3} {100 * ok / total:5.1f}%")
    if n:
        print(f"  {'TOTAL':9} {score:3}/{n:<3} {100 * score / n:5.1f}%   ({args.mode}, model {config.LLM_MODEL})")
    if ungraded:
        print(f"  not auto-graded (open answers, essays): {len(ungraded)}; read them in {out_path.name}")
    if args.mode == "base":
        print(f"  lenient (harness normalizer on raw output): {lenient_score}/{n} = {100 * lenient_score / max(n, 1):.1f}%")
    print(f"  time {elapsed:.0f}s, {elapsed / max(len(items), 1):.1f}s per question")
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
