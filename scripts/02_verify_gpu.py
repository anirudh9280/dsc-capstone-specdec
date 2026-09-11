"""M0 gate: prove this PyTorch build can actually drive an RTX 5080 (Blackwell, sm_120).

`torch.cuda.is_available()` returns True even for a wheel with no sm_120 kernels --
it only fails later, at kernel launch. So we check the compiled arch list AND run
real work with an explicit synchronize. Exits non-zero on failure; this gates M1.
"""
import sys

import torch

FAIL = []
REQUIRED_ARCH = "sm_120"  # RTX 5080 = GB203 = compute capability 12.0


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{': ' + detail if detail else ''}")
    if not ok:
        FAIL.append(label)
    return ok


print("=" * 62)
print("build")
print("=" * 62)
print(f"  torch          {torch.__version__}")
print(f"  cuda runtime   {torch.version.cuda}")
print(f"  arch list      {torch.cuda.get_arch_list()}")

print()
print("=" * 62)
print("device")
print("=" * 62)

if not check("cuda available", torch.cuda.is_available()):
    print("\nNo CUDA at all -- stop here.")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
free_b, total_b = torch.cuda.mem_get_info()

check("device name", True, name)
check("compute capability", (cc_major, cc_minor) == (12, 0), f"sm_{cc_major}{cc_minor}")
check(
    f"{REQUIRED_ARCH} in build",
    REQUIRED_ARCH in torch.cuda.get_arch_list(),
    "wheel lacks Blackwell kernels -- reinstall from a cu128+ index" 
    if REQUIRED_ARCH not in torch.cuda.get_arch_list() else "compiled in",
)
print(f"  [INFO] VRAM: {free_b / 1024**3:.2f} GiB free of {total_gb:.2f} GiB total")
if free_b / 1024**3 < 9.0:
    print("  [WARN] <9 GiB free. Close Overwatch/Chrome/Ollama before benchmarking.")

print()
print("=" * 62)
print("real work (the part is_available() cannot fake)")
print("=" * 62)

try:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    check("bf16 matmul 4096^3", torch.isfinite(c).all().item())
except Exception as e:  # noqa: BLE001
    check("bf16 matmul 4096^3", False, f"{type(e).__name__}: {e}")

try:
    q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    check("causal SDPA (transformers' attention path)", torch.isfinite(o).all().item())
except Exception as e:  # noqa: BLE001
    check("causal SDPA (transformers' attention path)", False, f"{type(e).__name__}: {e}")

# Achieved HBM bandwidth -- the denominator of every roofline number in this project.
# RTX 5080: 256-bit GDDR7 @ 30 Gbps = 960 GB/s theoretical.
try:
    import time

    buf = torch.empty(512 * 1024 * 1024 // 2, device="cuda", dtype=torch.bfloat16)  # 512 MiB
    for _ in range(3):
        buf.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        buf.clone()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    moved_gb = (buf.numel() * buf.element_size() * 2 * iters) / 1e9  # read + write
    bw = moved_gb / dt
    print(f"  [INFO] achieved copy bandwidth: {bw:.0f} GB/s ({bw / 960 * 100:.0f}% of 960 GB/s spec)")
except Exception as e:  # noqa: BLE001
    print(f"  [WARN] bandwidth probe failed: {type(e).__name__}: {e}")

print()
if FAIL:
    print(f"GATE FAILED: {', '.join(FAIL)}")
    sys.exit(1)
print("GATE PASSED -- environment is ready for M1.")
