#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
if ! python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio; then
  python -m pip install torch torchvision torchaudio
fi
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
PY
