# Running the model on a GPU

The laptop CPU needs 30-100 s per closed question and 9-14 min per essay; the exam and the
presentation have a few minutes on stage. The model therefore runs on a rented GPU, while the
harness, the knowledge base and the organisers' exam script stay on the laptop. The rules allow
this: the ban concerns external AI APIs and internet knowledge, not where our own model runs.

```
exam script -> harness (laptop, :8000) -> SSH tunnel -> Ollama (GPU machine, :11434) -> model
```

## 1. Start a GPU machine (Forgehand / Labqoat, sponsor credits)

```bash
fh classes                                   # gpu-l40s-small: 1x L40S 48 GB, about 1.86 USD/h
fh workspace create niedzielny-maturzysta matura
fh session start matura --class gpu-l40s-small --wait
fh session ls                                # note the session id
```

Billing runs from start to stop: stop the session after the exam (`fh session stop <id>`).

## 2. Serve the model

```bash
fh session ssh <id>
# on the GPU machine:
git clone https://github.com/romanszlazakai-lgtm/bielik-harness-matura.git
bash bielik-harness-matura/scripts/gpu_setup.sh              # Bielik 1.5B Q4_K_M
WITH_GEMMA=1 bash bielik-harness-matura/scripts/gpu_setup.sh # optionally also Gemma 4 12B QAT
```

The script ends with a one-question smoke test per model and prints its time.

## 3. Connect the laptop

In a second terminal on the laptop, keep the tunnel open for the whole exam:

```bash
fh session ssh <id> -- -N -L 11434:localhost:11434
```

Then point the harness at it and check with a few dev questions:

```powershell
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL = "bielik-1.5b-q4km"
$env:PROMPT_VERSION = "v3"
$env:USE_CKE = "1"
python scripts/run_dev.py --mode harness --limit 5
```

Optional essay model (read the note below first):

```powershell
$env:ESSAY_LLM_MODEL = "gemma-4-12b-qat"
```

## 4. On stage

```bash
python -m harness.server --port 8000        # exam script -> http://localhost:8000/v1
# or, for a sheet delivered as a file (PDF scans are OCR'd):
python main.py --questions arkusz.pdf --out odpowiedzi.json
```

The bare-model run points the exam script at `http://localhost:11434/v1` with model
`bielik-1.5b-q4km`.

## Note on using a second model

The rules say that with several models the progress baseline is the one that scores highest
alone, and "Maly, ale wariat" rewards the smallest model. Sending essays to Gemma 4 12B therefore
makes Gemma the baseline and the submission no longer "small". Decide the category first:
Bielik alone for smallest model and biggest progress, or Gemma for everything for best score.
