"""Runtime settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default):
    return os.environ.get(name, default)


# Model endpoint (OpenAI-compatible). LM Studio serves on port 1234 by default.
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = _env("LLM_MODEL", "bielik-1.5b-q4km")          # final model: Bielik 1.5B Q4_K_M (0.97 GB)
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "600"))  # long CKE sources take >3 min to read on a laptop CPU

# Harness behaviour.
VOTES = int(_env("VOTES", "1"))                      # answers per question; 3 = greedy + 2 samples, then vote
VOTE_TEMPERATURE = float(_env("VOTE_TEMPERATURE", "0.7"))
REASONING = _env("REASONING", "1") == "1"            # short fact recall before the answer
MAX_TOKENS = int(_env("MAX_TOKENS", "160"))
OPEN_MAX_TOKENS = int(_env("OPEN_MAX_TOKENS", "260"))    # open answers: answer + justification
OPEN_MAX_SENTENCES = int(_env("OPEN_MAX_SENTENCES", "2"))          # later sentences were mostly invented
DECISION_REASON_SENTENCES = int(_env("DECISION_REASON_SENTENCES", "1"))
ESSAY_MAX_TOKENS = int(_env("ESSAY_MAX_TOKENS", "1000"))  # one essay draft
ESSAY_MIN_WORDS = int(_env("ESSAY_MIN_WORDS", "300"))     # organisers' minimum; shorter drafts get extended
ESSAY_MAX_EXTENSIONS = int(_env("ESSAY_MAX_EXTENSIONS", "3"))
# Optional separate model for essays (e.g. a larger one on a GPU server). Empty = same as LLM_*.
# Note the rules: with several models, the progress baseline is the best of them used alone.
ESSAY_LLM_MODEL = _env("ESSAY_LLM_MODEL", "")
ESSAY_LLM_BASE_URL = _env("ESSAY_LLM_BASE_URL", "")
ESSAY_TIMEOUT = float(_env("ESSAY_TIMEOUT", "900"))       # a 1000-token draft takes minutes on a laptop CPU
USE_RAG = _env("USE_RAG", "1") == "1"
TOP_K = int(_env("TOP_K", "3"))                      # knowledge-base passages per question
PASSAGE_CHARS = int(_env("PASSAGE_CHARS", "600"))    # passage length cap inside the prompt
DECIMAL_SEPARATOR = _env("DECIMAL_SEPARATOR", ",")   # numeric answers: "4,5" or "4.5"

PROMPT_VERSION = _env("PROMPT_VERSION", "v3")          # v3: one line per item + item hints for list questions
# Worked examples from CKE papers before every question. Off by default: on the 1.5B model they
# lengthened the prompt to ~7000 characters, broke order and short answers, and doubled the time.
CKE_SHOTS = _env("CKE_SHOTS", "0") == "1"
USE_CKE = _env("USE_CKE", "1") == "1"                  # one passage slot for CKE papers and keys
CKE_MIN_COVERAGE = float(_env("CKE_MIN_COVERAGE", "0.5"))      # share of question words found in a CKE passage
CKE_MIN_INSTRUCTION = float(_env("CKE_MIN_INSTRUCTION", "0.6"))  # share of the CKE instruction found in the question

KB_DIR = Path(_env("KB_DIR", str(ROOT / "data" / "kb")))
CKE_DIR = ROOT / "data" / "cke"
KB_EXTRA_DIR = ROOT / "data" / "kb_extra"
OUTPUT_DIR = ROOT / "outputs"
