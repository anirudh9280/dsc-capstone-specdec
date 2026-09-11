#!/usr/bin/env bash
# M0: Install Miniconda into WSL ext4 home. No sudo required.
set -euo pipefail

PREFIX="$HOME/miniconda3"

if [ -x "$PREFIX/bin/conda" ]; then
  echo "[skip] conda already at $PREFIX"
  "$PREFIX/bin/conda" --version
  exit 0
fi

INSTALLER="/tmp/miniconda-installer.sh"
URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

echo "[1/3] downloading Miniconda..."
curl -fsSL "$URL" -o "$INSTALLER"

echo "[2/3] installing to $PREFIX (batch mode)..."
bash "$INSTALLER" -b -p "$PREFIX"

echo "[3/3] initializing shell hook..."
"$PREFIX/bin/conda" init bash >/dev/null
"$PREFIX/bin/conda" config --set auto_activate_base false

rm -f "$INSTALLER"
echo "[done] $("$PREFIX/bin/conda" --version)"
