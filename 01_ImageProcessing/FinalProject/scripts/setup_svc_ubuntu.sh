#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SVC_DIR="${PROJECT_DIR}/stable-virtual-camera"
NATIVE_SVC_DIR="${HOME}/.cache/imageprocessing-svc/stable-virtual-camera"
VENV_DIR="${HOME}/.cache/imageprocessing-svc/venv"
VENV_PATH_FILE="${SVC_DIR}/.svc_venv_path"
SVC_PATH_FILE="${SVC_DIR}/.svc_repo_path"
LEGACY_VENV_PATH_FILE="${SVC_DIR}/.wsl_svc_venv_path"
LEGACY_SVC_PATH_FILE="${SVC_DIR}/.wsl_svc_repo_path"

if [[ ! -f "${SVC_DIR}/demo.py" ]]; then
  echo "Stable Virtual Camera repo was not found at: ${SVC_DIR}" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi is not visible. Check the NVIDIA driver/CUDA setup before generating views." >&2
else
  nvidia-smi
fi

if ! command -v python3 >/dev/null 2>&1 && ! command -v python3.10 >/dev/null 2>&1 && ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python is not installed. Run this first, then retry:" >&2
  echo "  sudo apt update" >&2
  echo "  sudo apt install -y python3 python3-venv python3-pip git build-essential" >&2
  exit 1
fi

PYTHON_BIN=""
for candidate in python3.10 python3.11 python3; do
  if ! command -v "${candidate}" >/dev/null 2>&1; then
    continue
  fi

  test_dir="$(mktemp -d)"
  if "${candidate}" -m venv "${test_dir}/venv" >/dev/null 2>&1; then
    PYTHON_BIN="${candidate}"
    rm -rf "${test_dir}"
    break
  fi
  rm -rf "${test_dir}"
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No Python installation can create a virtual environment." >&2
  echo "Run this, then retry:" >&2
  echo "  sudo apt update" >&2
  echo "  sudo apt install -y python3 python3-venv python3-pip git build-essential" >&2
  echo "If you installed a version-specific Python such as python3.10, also run:" >&2
  echo "  sudo apt install -y python3.10-venv" >&2
  exit 1
fi

echo "Using Python for SVC venv: ${PYTHON_BIN}"

cd "${SVC_DIR}"
if ! git status >/dev/null 2>&1; then
  git config --global --add safe.directory "${SVC_DIR}"
fi
if ! git submodule update --init --recursive; then
  echo "Warning: git submodule update failed." >&2
  echo "Continuing because the single-image SVC CLI path used by this project does not require the dust3r submodule." >&2
fi

rm -rf "${NATIVE_SVC_DIR}"
mkdir -p "${NATIVE_SVC_DIR}"
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.venv-wsl' \
  --exclude='.svc_repo_path' \
  --exclude='.svc_venv_path' \
  --exclude='.wsl_svc_repo_path' \
  --exclude='.wsl_svc_venv_path' \
  --exclude='work_dirs' \
  --exclude='seva.egg-info' \
  -cf - -C "${SVC_DIR}" . | tar -xf - -C "${NATIVE_SVC_DIR}"
printf "%s\n" "${NATIVE_SVC_DIR}" > "${SVC_PATH_FILE}"
printf "%s\n" "${NATIVE_SVC_DIR}" > "${LEGACY_SVC_PATH_FILE}"

mkdir -p "$(dirname "${VENV_DIR}")"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
printf "%s\n" "${VENV_DIR}" > "${VENV_PATH_FILE}"
printf "%s\n" "${VENV_DIR}" > "${LEGACY_VENV_PATH_FILE}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6" torchvision torchaudio
cd "${NATIVE_SVC_DIR}"
python -m pip install -e .
python -m pip install scipy
python -m pip install "git+https://github.com/jensenz-sai/pycolmap@543266bc316df2fe407b3a33d454b310b1641042"

python - <<'PY'
import importlib.util
import torch

missing = [m for m in ("fire", "seva", "torch", "pycolmap", "scipy") if importlib.util.find_spec(m) is None]
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
if missing:
    raise SystemExit(f"Missing modules after install: {missing}")
PY

cat <<EOF

SVC Ubuntu/Linux environment is ready.

Next:
  cd ${PROJECT_DIR}
  python3 src/generate_views_svc.py --backend native --image data/source/Jeans.png --out data/raw/Jeans_svc --target-count 40

The wrapper will automatically read:
  ${SVC_PATH_FILE}
  ${VENV_PATH_FILE}
EOF
