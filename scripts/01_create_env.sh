#!/usr/bin/env bash
# M0: Create the `specdec` conda env on ext4 with Blackwell-capable PyTorch.
set -euo pipefail

CONDA="$HOME/miniconda3/bin/conda"
ENV_NAME="specdec"
PYVER="3.11"
CUDA_CHANNEL="${CUDA_CHANNEL:-cu128}"   # override: CUDA_CHANNEL=cu129 bash 01_create_env.sh

# Keep model weights and big traces on ext4, NEVER on /mnt/c (drvfs is far too slow
# for multi-GB safetensors, and results/ must not land in OneDrive sync).
HF_HOME_DIR="$HOME/.cache/huggingface"
RESULTS_DIR="$HOME/specdec-results"

if ! "$CONDA" env list | grep -qE "^${ENV_NAME}\s"; then
  echo "[1/4] creating env ${ENV_NAME} (python ${PYVER})..."
  "$CONDA" create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=${PYVER}" pip
else
  echo "[1/4] env ${ENV_NAME} already exists, reusing"
fi

ENV_PREFIX="$("$CONDA" env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
PIP="$ENV_PREFIX/bin/pip"

echo "[2/4] installing PyTorch (${CUDA_CHANNEL} wheels, needs sm_120 for RTX 5080)..."
"$PIP" install --upgrade pip
"$PIP" install torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

echo "[3/4] installing research stack..."
"$PIP" install \
  "transformers>=4.44" accelerate datasets huggingface_hub safetensors \
  sentencepiece tokenizers \
  numpy pandas scipy scikit-learn matplotlib \
  tqdm rich pytest

echo "[4/4] pinning env vars (HF cache + results on ext4)..."
mkdir -p "$HF_HOME_DIR" "$RESULTS_DIR"
"$CONDA" env config vars set -n "$ENV_NAME" \
  HF_HOME="$HF_HOME_DIR" \
  SPECDEC_RESULTS="$RESULTS_DIR" \
  TOKENIZERS_PARALLELISM=false

echo
echo "[done] activate with:  conda activate ${ENV_NAME}"
echo "       env prefix:     ${ENV_PREFIX}"
echo "       HF_HOME:        ${HF_HOME_DIR}"
echo "       results:        ${RESULTS_DIR}"
