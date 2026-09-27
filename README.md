# Matura harness for a small model

A harness that lets a 1.5B-parameter Polish model sit the Polish history matura. Built for the
"Maly, ale wariat" category of the Warsaw Model Trainers hackathon: the smallest model that scores
at least 35%.

- **Model:** Bielik-1.5B-v3.0-Instruct, unchanged (no fine-tuning, no adapters), quantised to
  **Q4_K_M, 0.97 GB**: `Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf` from
  [second-state/Bielik-1.5B-v3.0-Instruct-GGUF](https://huggingface.co/second-state/Bielik-1.5B-v3.0-Instruct-GGUF).
  One model answers the whole sheet, essays included.
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

**Final configuration:** Bielik-1.5B-v3.0-Instruct **Q4_K_M, 0.97 GB**, harness v3 (one line per
item for list questions, item hints, CKE task lookup), one answer per question. These are the
defaults in `harness/config.py`.

Two test sets. **Dev:** 46 closed questions written by the team (`data/dev/`), optimistic because
the cheat sheet was written by the same team. **Held-out CKE:** the 10 closed items of the May
2023 paper that fit the exam's formats, kept out of every index; the honest check. "Strict" means
the reply must be exactly the expected answer.

| Configuration | Dev (46) | Held-out CKE (10) |
|---|---|---|
| Bare Q4_K_M, strict | 3 (6.5%) | 1 (10%) |
| Bare Q4_K_M, our normalizer on its replies | 16 (34.8%) | 2 (20%) |
| Harness v1: prompts, retrieval, 3 votes | 28 (60.9%) | |
| Harness v2: code assembles facts, no votes | 31 (67.4%) | 4 (40%) |
| v3 with 3 CKE worked examples in every prompt | 33 (71.7%) | 2 (20%) |
| **v3 final: list templates + item hints, no CKE examples** | **35 (76.1%)** | **4 (40%)** |

**How small can the model get?** Same harness (v3 final), same sets:

| Quantisation | Size | Dev (46) | Held-out CKE (10) | Verdict |
|---|---|---|---|---|
| Q8_0 (harness v2) | 1.70 GB | 35 (76.1%) | | same accuracy, 1.8x the size |
| **Q4_K_M** | **0.97 GB** | **35 (76.1%)** | **4 (40%)** | **chosen** |
| Q3_K_M | 0.78 GB | 25 (54.3%) | 4 (40%), 2 lost to timeouts | weaker on every other test; essay under 300 words |
| Q2_K | 0.61 GB | 6 of the first 18 | 0 (0%) | breaks: answers the prompt's example ("Kircholm") instead of the question |

**Essays and open answers** (Q4_K_M, final version; not auto-graded, read by the team): both test
essays above 300 words (314 and 433), plain prose, structure kept, dates taken from the timeline;
both "Rozstrzygnij" decisions correct after asking for facts before the verdict. Content remains
the weak point: the 1.5B model still invents details beyond the first sentence, which is why open
answers are cut to their first sentences.

**Scanned PDF sheets:** the May 2023 paper OCR'd as if it were a scan (36 pages, 6.8 s per page on
the laptop): 37 of 37 tasks found, all with the same type as from the text layer.

**Speed:** on the laptop CPU (i5-6300U) prompt reading runs at 6-8 tokens/s: 30-90 s per closed
question and 5-14 min per essay. The exam runs on a GPU (`RUN_GPU.md`); model and harness are
unchanged.

### What each version changed, and why

- **v1 → v2** (the model knew the facts but assembled them wrongly): order questions: the model
  writes a year per letter, code sorts. Matching: the element is mapped to the category text the
  model wrote. Multi: one TAK/NIE per option. Format examples such as "np. P,F,P" are removed from
  the question, because the model copied them. Voting off: it fixed one answer and broke another
  while doubling the time.
- **v2 → v3:** one line per item with a fact hint retrieved for each statement, option or element;
  a time criterion in a multi question ("w XIX wieku") is checked in code against the years the
  model wrote. Three CKE worked examples in every prompt made the prompt ~7000 characters long and
  the 1.5B model lost track (order 4/4 → 1/4, held-out 4 → 2), so they are off.
- **Essays and open answers:** own type, structured prompt, word count and extension in code,
  markdown and invented references removed, justifications cut after the first sentence(s).

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
- **List questions in v3** (true/false, multi, matching) are answered one line per item:
  `1. fact => P`, `A: fact with a date => TAK`, `element => category text`. Every item also
  gets its own fact hint from the knowledge base, because a single search for the whole question
  only covers its dominant topic. The parser reads each line's verdict, maps matching lines by
  word stems (so an inflected name such as "Bitwy pod Wiedniem" still matches), and checks a time
  criterion in the question ("w XIX wieku", "w 1989 r.") against the year the model wrote
  instead of trusting the model's TAK/NIE.
- **`--holdout`** keeps one exam out of the index and writes its closed tasks to
  `data/cke/heldout.jsonl` in the dev-set format, for an honest check on real CKE questions.

## Open answers and essays

- **Detection.** A question is an `essay` when it asks for a *wypracowanie*, *rozprawka* or a
  minimum word count, or offers topics to choose from (checked before single choice, because
  topics are often labelled A/B/C). It is `open` when it asks for reasons (*wyjaśnij, uzasadnij,
  rozstrzygnij, porównaj, oceń, przedstaw...*), and `short` only with an explicit "jak najkrócej".
- **Open answers** (up to 260 tokens) start with the answer, then a concrete fact. Tasks that say
  *Rozstrzygnij* answer in the marking scheme's own shape: `Rozstrzygnięcie:` then `Uzasadnienie:`.
- **Essays** (`harness/essay.py`): with several topics, the one the knowledge base covers best is
  chosen and named on the first line. Five passages of context, then one draft of up to 1000
  tokens in a fixed structure: Wstęp (thesis and time frame), Argument 1-3 (fact, analysis,
  conclusion), Podsumowanie. Code then counts the words (repeated sentences excluded). Under 300
  words, or without a conclusion, the model is asked to add a further argument and a conclusion,
  at most twice. Section labels, repeated sentences and a half-sentence cut off by the token limit
  are removed from the final text.
- **Grading.** These answers are not auto-graded: the dev runner saves them for human review.
  `scripts/test_essay.py` checks detection, prompts and the extension loop with a stub model.

## Reproduce

```bash
# 1. Model: download a GGUF above (LM Studio: search "Bielik-1.5B-v3.0-Instruct"),
#    load it as "bielik-1.5b-q4km" and start the server on port 1234. Or with curl:
curl -L -o Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf https://huggingface.co/second-state/Bielik-1.5B-v3.0-Instruct-GGUF/resolve/main/Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf

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
