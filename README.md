# Matura harness for a small model

A harness that lets a 1.5B-parameter Polish model sit the Polish history matura. Built for the
"Maly, ale wariat" category of the Warsaw Model Trainers hackathon: the smallest model that scores
at least 35%.

- **Model:** Bielik-1.5B-v3.0-Instruct, unchanged (no fine-tuning, no adapters).
  - Q8_0, 1.70 GB: [speakleash/Bielik-1.5B-v3.0-Instruct-GGUF](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct-GGUF)
  - Q4_K_M, 0.97 GB: [second-state/Bielik-1.5B-v3.0-Instruct-GGUF](https://huggingface.co/second-state/Bielik-1.5B-v3.0-Instruct-GGUF)
- **Runs on a laptop CPU** (Intel i5-6300U, 16 GB RAM, no GPU) through LM Studio, or any
  OpenAI-compatible server such as llama.cpp.
- **Offline during the exam.** Internet is used only once, to build the knowledge base.
- **Standard library only.** Python 3.10+, no `pip install`.

## How it works

```
exam script -> harness server (looks like a model: OpenAI or Ollama API)
                 1. detect the answer type from the question wording
                    (single, multi, true/false list, matching, order, numeric, short)
                 2. retrieve 3 passages from the local knowledge base with BM25 (no extra model):
                    the best cheat-sheet passage first, then Wikipedia, one passage per article
                 3. prompt with one worked example for that type, short fact recall first
                 4. one greedy answer by default (VOTES=3 adds 2 samples at temperature 0.7 and votes)
                 5. normalize each answer to the exact format the grader expects
                 6. vote: whole-answer majority, or per position for P/F lists and matchings
             -> Bielik 1.5B
```

The small model does not have to remember history: the knowledge base holds the facts, and the
model mostly reads and chooses. The normalizer turns a rambling reply into a well-formed answer,
and a closed question is never left blank.

| Knowledge base | |
|---|---|
| Polish Wikipedia | 90 seed articles (`data/kb_seeds.txt`) and the articles they link to most, 2461 articles |
| Cheat sheet | `data/kb_extra/chronologia.md`: chronology of Polish and world history, art styles, key paintings |
| Index | about 56 000 passages of ~120 words, BM25 with 6-letter prefix stemming for Polish inflection |

## Results

Dev set: 46 questions, see `data/dev/README.md`. "Strict" means the reply must be exactly the
expected answer; "lenient" applies this harness's normalizer to the bare model's reply.

| Model | Bare, strict | Bare, lenient | Harness v1 (3 votes) | Harness v1 replies, v2 normalizer |
|---|---|---|---|---|
| Bielik 1.5B Q8_0 (1.70 GB) | 3/46 (6.5%) | 18/46 (39.1%) | 29/46 (63.0%) | 34/46 (73.9%) |
| Bielik 1.5B Q4_K_M (0.97 GB) | 3/46 (6.5%) | 16/46 (34.8%) | 28/46 (60.9%) | 28/46 (60.9%) |

Harness v2 run (1 answer per question): Q4_K_M 31/46 (67.4%), 32.3 s per question.

Seconds per question on the laptop CPU: bare model 29.7 (Q8) and 18.4 (Q4); harness v1 with 3 votes
98.2 (Q8) and 71.1 (Q4).

On the laptop CPU a question takes 70-100 s (prompt reading runs at about 7.5 tokens/s), so the
exam should run on a GPU; the model and harness are unchanged either way.

What v2 changed, after reading the v1 replies (the model often knew the facts but assembled them
wrongly):

- **order:** the model writes a year or century per letter and the code sorts them.
- **matching:** each element is matched to the category whose text the model wrote next to it.
- **multi:** one TAK/NIE verdict per option instead of a bare list of letters.
- **tflist and order:** answer-format examples such as "np. P,F,P" are removed from the question,
  because the small model copied them as its answer.
- **numeric:** year differences are never negative.
- **voting:** off by default. Across the whole v1 run it fixed one answer and broke another, while
  doubling the time.

## CKE papers (v3, optional)

`scripts/build_cke.py` turns CKE exam papers, their marking schemes and exam materials into a local
knowledge base: 70 tasks from the March 2022 sample paper and the May 2023 paper, each with its
answer from the marking scheme, mapped onto the exam's formats where possible (6 single choice,
5 true/false, 2 matching, 14 short, 16 decisions, 27 open). It reads PDFs through `pdftotext` and
OCR exports in RTF (`harness/rtf.py`). CKE papers quote copyrighted sources, so everything it
builds stays in `data/cke/` and is not committed; the papers are public on cke.gov.pl.

```bash
python scripts/build_cke.py --src path/to/cke/papers                    # for the exam
python scripts/build_cke.py --src path/to/cke/papers --holdout 2023-05  # for evaluation
PROMPT_VERSION=v3 USE_CKE=1 python scripts/run_dev.py --mode harness --data data/cke/heldout.jsonl
```

- **`PROMPT_VERSION=v3`** prepends three worked examples from the 2022 sample paper, listed in
  `data/cke_fewshot.json` with the team's reasoning. They are the same prefix for every question,
  so the model server reads them once and then serves them from its prompt cache.
- **`USE_CKE=1`** recognises a question that repeats a CKE task and puts that task's marking-scheme
  answer into the context. Two conditions must hold: most of the question's words occur in one
  passage, and most of the CKE task's own instruction occurs in the question. On the 2023 paper
  this finds the right subtask for 10 of 10 closed items. On the 46 dev questions, which repeat no
  CKE task, it fires 0 times.
- **`--holdout`** keeps one exam out of the index and writes its closed tasks to
  `data/cke/heldout.jsonl` in the dev-set format, for an honest check on real CKE questions.

## Reproduce

```bash
# 1. Model: download a GGUF above (LM Studio: search "Bielik-1.5B-v3.0-Instruct"),
#    load it and start the server on port 1234. Or with curl:
curl -L -o Bielik-1.5B-v3.0-Instruct.Q8_0.gguf https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct-GGUF/resolve/main/Bielik-1.5B-v3.0-Instruct.Q8_0.gguf

# 2. Knowledge base (needs internet once; Wikipedia rate-limits, allow 20-40 min):
python scripts/build_kb.py

# 3. Checks without a model:
python scripts/test_normalize.py

# 4. Dev set, bare model vs harness:
python scripts/run_dev.py --mode base
python scripts/run_dev.py --mode harness
```

Settings are environment variables (see `harness/config.py`): `LLM_BASE_URL`, `LLM_MODEL`,
`VOTES`, `REASONING`, `USE_RAG`, `TOP_K`, `DECIMAL_SEPARATOR`.

## On stage

```bash
# LM Studio server running with the model loaded, then:
python -m harness.server --port 8000
```

Point the exam script at `http://localhost:8000/v1` (OpenAI style) or `http://localhost:8000`
(Ollama style). Every question and answer is logged to `outputs/server_log.jsonl`.
The bare-model run points the exam script straight at LM Studio (`http://localhost:1234/v1`).

If the exam comes as a file instead, `main.py` answers it with the same pipeline:

```bash
python main.py --questions exam.json --out answers.json               # harness
python main.py --questions exam.json --out base.json --mode base      # untouched model
python main.py --questions exam.json --out answers.json --resume      # continue after a crash
```

Input is `{"questions": [...]}`, a JSON list or JSON Lines, with `id`, `question`, and optionally
`type` and `options`. `--format` picks the output: `answers` (default,
`{"answers": [{"question_id", "given"}]}`), `map`, `jsonl` or `csv`. Answers are written after
every question, and a per-question trace goes to `outputs/`.

## Sources

| Source | Licence | How it is used |
|---|---|---|
| Polish Wikipedia, seed list in `data/kb_seeds.txt` plus their most-linked articles | CC BY-SA 4.0 | Fetched by `scripts/build_kb.py` into `data/kb/` (not committed) |
| `data/kb_extra/chronologia.md` | written by the team | Chronology and art-style cheat sheet |
| `data/dev/historia_dev.jsonl` | written by the team (LLM-assisted) | 46 dev questions in the exam's answer formats |
| Bielik-1.5B-v3.0-Instruct | Apache 2.0 | The model, unchanged |

No exam papers or other copyrighted material are stored in this repository. A knowledge base built
from Wikipedia is a derived dataset and falls under CC BY-SA.
