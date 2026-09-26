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
VOTES = int(_env("VOTES", "3"))                      # answers sampled per question
VOTE_TEMPERATURE = float(_env("VOTE_TEMPERATURE", "0.7"))
REASONING = _env("REASONING", "1") == "1"            # short fact recall before the answer
MAX_TOKENS = int(_env("MAX_TOKENS", "160"))
USE_RAG = _env("USE_RAG", "1") == "1"
TOP_K = int(_env("TOP_K", "3"))                      # knowledge-base passages per question
PASSAGE_CHARS = int(_env("PASSAGE_CHARS", "600"))    # passage length cap inside the prompt
DECIMAL_SEPARATOR = _env("DECIMAL_SEPARATOR", ",")   # numeric answers: "4,5" or "4.5"

KB_DIR = Path(_env("KB_DIR", str(ROOT / "data" / "kb")))
KB_EXTRA_DIR = ROOT / "data" / "kb_extra"
OUTPUT_DIR = ROOT / "outputs"
