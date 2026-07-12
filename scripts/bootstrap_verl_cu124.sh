#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-/data/zhangdw12/venvs/verl-cu124}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
BFCL_SOURCE="${BFCL_SOURCE:-}"
BFCL_WHEEL="${BFCL_WHEEL:-}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: the CUDA 12.4 lock targets Linux x86_64" >&2
  exit 1
fi

command -v uv >/dev/null || {
  echo "error: uv is required" >&2
  exit 1
}
command -v nvidia-smi >/dev/null || {
  echo "error: nvidia-smi is required" >&2
  exit 1
}

echo "== NVIDIA driver =="
nvidia-smi
if command -v nvcc >/dev/null; then
  echo "== CUDA toolkit =="
  nvcc --version
fi

uv venv --python "$PYTHON_VERSION" "$VENV_PATH"
PYTHON="$VENV_PATH/bin/python"

uv pip install \
  --python "$PYTHON" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r "$REPO_ROOT/requirements-cu124.lock"

uv pip install --python "$PYTHON" --no-deps -e "$REPO_ROOT"

if [[ -n "$BFCL_SOURCE" ]]; then
  BFCL_SOURCE="$(cd "$BFCL_SOURCE" && pwd)"
  uv pip install --python "$PYTHON" --no-deps -e "$BFCL_SOURCE"
elif [[ -n "$BFCL_WHEEL" ]]; then
  uv pip install --python "$PYTHON" --no-deps "$BFCL_WHEEL"
else
  echo "error: set BFCL_SOURCE or BFCL_WHEEL so bfcl_eval is installed in the VeRL environment" >&2
  exit 1
fi

uv pip check --python "$PYTHON"

"$PYTHON" - <<'PY'
import importlib

import torch

required = [
    "vllm",
    "flash_attn",
    "flashinfer",
    "transfer_queue",
    "verl",
    "bfcl_eval",
    "bfcl_eval.constants.executable_backend_config",
    "bfcl_eval.constants.default_prompts",
    "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker",
    "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils",
]

for module in required:
    importlib.import_module(module)

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if torch.version.cuda != "12.4":
    raise SystemExit(f"expected torch CUDA 12.4, got {torch.version.cuda}")

print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("GPU", torch.cuda.get_device_name(0))
print("VeRL CUDA 12.4 environment is ready")
PY
