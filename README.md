# Speculative Decoding — DSC Capstone D38

Efficient AI (Prof. Zhijian Liu, HDSI). Track: **inference-time acceleration via
speculative decoding**. Goal for the first meeting: acceptance length on MATH-500,
plus a measured account of where the headroom actually is.

## Hardware / environment

| | |
|---|---|
| GPU | RTX 5080, 16 GB, Blackwell **sm_120** |
| Host | Ryzen 9, Windows 11 |
| Runtime | **WSL2 Ubuntu 24.04**, conda env `specdec` (python 3.11) |
| Weights + traces | WSL **ext4** (`$HF_HOME`, `$SPECDEC_RESULTS`) — never `/mnt/c` |

Two environment facts that cost real time if forgotten:

1. **sm_120 needs cu128+ PyTorch wheels.** A wheel without Blackwell kernels
   imports fine and `torch.cuda.is_available()` returns `True`; it dies at kernel
   launch. `scripts/02_verify_gpu.py` gates on `get_arch_list()` plus real work.
2. **Never put the HF cache or a conda env on `/mnt/c`.** drvfs makes multi-GB
   safetensors loads pathologically slow.

## Setup

```bash
bash scripts/00_install_miniconda.sh    # installs to ~/miniconda3, no sudo
bash scripts/01_create_env.sh           # conda-forge only (avoids Anaconda ToS)
conda activate specdec
python scripts/02_verify_gpu.py         # must print GATE PASSED
```

## Layout

```
specdec/   timing.py    CUDA-correct timers (events, median+IQR)
           loop.py      from-scratch speculative decoding w/ explicit KV rollback
bench/     roofline.py  is decode memory-bound on this card?
analysis/               offline analysis over logged traces
tests/     test_lossless.py   greedy spec-dec == greedy autoregressive (hard gate)
notes/                  paper notes + writeup
```

## Ground rules for measurement

- `torch.cuda.synchronize()` (or CUDA events) around every timer — see `specdec/timing.py`.
- Median + IQR, never mean. Desktop GPUs have a heavy right tail.
- Close Ollama / Overwatch / Chrome before benchmarking; record what was closed.
- Speculative decoding is **lossless**. Greedy output must be token-identical to
  plain greedy decoding from the target. Any divergence is a cache bug, not a tolerance.
