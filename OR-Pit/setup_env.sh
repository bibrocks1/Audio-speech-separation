#!/usr/bin/env bash
# OR-PiT environment setup for Linux (CPU or CUDA). Usage: ORPIT_CUDA=cu124 ./setup_env.sh
set -euo pipefail
VENV=${ORPIT_VENV:-.venv}
CUDA=${ORPIT_CUDA:-cpu}
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install torch==2.6.0 --index-url "https://download.pytorch.org/whl/$CUDA" || "$VENV/bin/python" -m pip install torch==2.6.0
"$VENV/bin/python" -m pip install -r "$(dirname "$0")/requirements.txt"
"$VENV/bin/python" -c "import torch;print(torch.__version__, torch.cuda.is_available())"
