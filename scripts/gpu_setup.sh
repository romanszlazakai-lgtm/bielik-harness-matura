#!/usr/bin/env bash
# Run ON the GPU machine (Linux). Serves the model(s) through Ollama's OpenAI-compatible API on
# port 11434. The harness stays on the laptop and reaches this port through an SSH tunnel.
#
#   bash gpu_setup.sh                 # Bielik 1.5B Q4_K_M
#   WITH_GEMMA=1 bash gpu_setup.sh    # also Gemma 4 12B QAT (6.98 GB), e.g. for essays
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR"
export PATH="$HOME/ollama/bin:$PATH"

# 1. Ollama: system install when we can write to /usr/local, otherwise the standalone tarball.
if ! command -v ollama >/dev/null; then
  if [ "$(id -u)" = "0" ] || sudo -n true 2>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    mkdir -p "$HOME/ollama"
    curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C "$HOME/ollama"
  fi
fi
if ! curl -s localhost:11434/api/tags >/dev/null; then
  nohup ollama serve > "$MODELS_DIR/ollama.log" 2>&1 &
  for _ in $(seq 30); do curl -s localhost:11434/api/tags >/dev/null && break; sleep 1; done
fi

fetch() {  # fetch <url> <file>: resumable download into MODELS_DIR
  [ -s "$MODELS_DIR/$2" ] || curl -fL -C - -o "$MODELS_DIR/$2" "$1"
}

# 2. Bielik 1.5B, same file as on the laptop. The GGUF's chat template is ChatML; it is spelled out
#    here so the prompts match LM Studio exactly.
fetch https://huggingface.co/second-state/Bielik-1.5B-v3.0-Instruct-GGUF/resolve/main/Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf \
      Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf
cat > "$MODELS_DIR/Modelfile.bielik" <<EOF
FROM $MODELS_DIR/Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER num_ctx 4096
EOF
ollama create bielik-1.5b-q4km -f "$MODELS_DIR/Modelfile.bielik"

# 3. Optional: Gemma 4 12B, Google's quantisation-aware Q4_0 (text only, no vision projector).
if [ "${WITH_GEMMA:-0}" = "1" ]; then
  fetch https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf \
        gemma-4-12b-it-qat-q4_0.gguf
  printf 'FROM %s\nPARAMETER num_ctx 8192\n' "$MODELS_DIR/gemma-4-12b-it-qat-q4_0.gguf" > "$MODELS_DIR/Modelfile.gemma"
  ollama create gemma-4-12b-qat -f "$MODELS_DIR/Modelfile.gemma"
fi

# 4. Smoke test: one Polish question through the same API the harness uses.
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
for model in bielik-1.5b-q4km $([ "${WITH_GEMMA:-0}" = "1" ] && echo gemma-4-12b-qat); do
  start=$(date +%s%N)
  reply=$(curl -s localhost:11434/v1/chat/completions -H 'Content-Type: application/json' -d "{
    \"model\": \"$model\", \"temperature\": 0, \"max_tokens\": 60,
    \"messages\": [{\"role\": \"user\", \"content\": \"W którym roku odbyła się bitwa pod Grunwaldem? Odpowiedz jednym zdaniem.\"}]}")
  echo "$model ($(( ($(date +%s%N) - start) / 1000000 )) ms): $(echo "$reply" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')"
done
echo "Ready. On the laptop: fh session ssh <session> -- -N -L 11434:localhost:11434"
