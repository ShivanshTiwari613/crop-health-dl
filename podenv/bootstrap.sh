#!/usr/bin/env bash
# Runs on the pod. Builds a TensorFlow venv on the persistent /workspace volume
# so stop/start cycles never reinstall. Re-runs are fast no-ops.
set -euo pipefail

# Some community hosts pass nvidia-smi but fail cuInit; exit 42 so pod.py
# terminates this pod instead of stopping it (a stopped pod restarts on the
# same host).
python3 - <<'PY'
import ctypes, sys
rc = ctypes.CDLL("libcuda.so.1").cuInit(0)
if rc != 0:
    print(f"FATAL: broken GPU on this host (cuInit -> {rc})", file=sys.stderr)
    sys.exit(42)
print("GPU preflight ok")
PY

VENV=/workspace/venv-tf
if [ ! -f "$VENV/bin/activate" ]; then
  python -m venv "$VENV"
fi
source "$VENV/bin/activate"

# tensorflow[and-cuda] bundles its own CUDA libraries, so it only needs the
# host driver. Pin the exact version used on the Mac.
pip install --quiet --upgrade pip
pip install --quiet "tensorflow[and-cuda]==2.20.0" \
  "tensorflow-datasets>=4.9" importlib_resources \
  "scikit-learn>=1.4" "pyyaml>=6" "matplotlib>=3.8" "pandas>=2.2"

python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"bootstrap ok: tensorflow {tf.__version__}, gpus={[g.name for g in gpus]}")
if not gpus:
    raise SystemExit("no GPU visible to TensorFlow")
PY
