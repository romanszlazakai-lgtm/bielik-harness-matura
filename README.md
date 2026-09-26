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
| Bielik 1.5B Q4_K_M (0.97 GB) | pending | pending | pending | pending |

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

## Sources

| Source | Licence | How it is used |
|---|---|---|
| Polish Wikipedia, seed list in `data/kb_seeds.txt` plus their most-linked articles | CC BY-SA 4.0 | Fetched by `scripts/build_kb.py` into `data/kb/` (not committed) |
| `data/kb_extra/chronologia.md` | written by the team | Chronology and art-style cheat sheet |
| `data/dev/historia_dev.jsonl` | written by the team (LLM-assisted) | 46 dev questions in the exam's answer formats |
| Bielik-1.5B-v3.0-Instruct | Apache 2.0 | The model, unchanged |

No exam papers or other copyrighted material are stored in this repository. A knowledge base built
from Wikipedia is a derived dataset and falls under CC BY-SA.
