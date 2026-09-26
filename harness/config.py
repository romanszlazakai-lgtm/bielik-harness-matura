"""Runtime settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default):
    return os.environ.get(name, default)


# Model endpoint (OpenAI-compatible). LM Studio serves on port 1234 by default.
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = _env("LLM_MODEL", "bielik-1.5b-v3.0-instruct")
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "180"))

# Harness behaviour.
VOTES = int(_env("VOTES", "1"))                      # answers per question; 3 = greedy + 2 samples, then vote
VOTE_TEMPERATURE = float(_env("VOTE_TEMPERATURE", "0.7"))
REASONING = _env("REASONING", "1") == "1"            # short fact recall before the answer
MAX_TOKENS = int(_env("MAX_TOKENS", "160"))
USE_RAG = _env("USE_RAG", "1") == "1"
TOP_K = int(_env("TOP_K", "3"))                      # knowledge-base passages per question
PASSAGE_CHARS = int(_env("PASSAGE_CHARS", "600"))    # passage length cap inside the prompt
DECIMAL_SEPARATOR = _env("DECIMAL_SEPARATOR", ",")   # numeric answers: "4,5" or "4.5"

PROMPT_VERSION = _env("PROMPT_VERSION", "v2")          # v3 adds worked examples from CKE papers
USE_CKE = _env("USE_CKE", "0") == "1"                  # one passage slot for CKE papers and keys
CKE_MIN_COVERAGE = float(_env("CKE_MIN_COVERAGE", "0.5"))      # share of question words found in a CKE passage
CKE_MIN_INSTRUCTION = float(_env("CKE_MIN_INSTRUCTION", "0.6"))  # share of the CKE instruction found in the question

KB_DIR = Path(_env("KB_DIR", str(ROOT / "data" / "kb")))
CKE_DIR = ROOT / "data" / "cke"
KB_EXTRA_DIR = ROOT / "data" / "kb_extra"
OUTPUT_DIR = ROOT / "outputs"
