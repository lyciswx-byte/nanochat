"""Single-launch cooperative NVFP4 2D activation quantizer for GB10.

The output has the exact tensor shapes, layouts, and e4m3 block scale policy
of torchao 0.19's *plain* triton_weight_quantize_2d. An in-launch device-wide
amax reduction uses grid.sync. Reading input a second time is unavoidable
for an exact global amax absent a prohibitively large residency/staging scheme.

No tensors survive a call in a Python cache. The compiled extension contains
only code; its PyTorch outputs are freshly allocated and may be donated safely.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
import torch


@lru_cache(maxsize=1)
def _extension():
    # IMPORTANT: build before torch.compile / CUDA Graph capture (prewarm below).
    from torch.utils.cpp_extension import load
    source = Path(__file__).resolve().parent / "csrc" / "fused_quant.cu"
    return load(
        name="nanochat_sm121_fused_quant_v1",
        sources=[str(source)],
        verbose=os.environ.get("NANOCHAT_FUSED_VERBOSE", "0") == "1",
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", "--fmad=false",
            "-gencode=arch=compute_121a,code=sm_121a",
        ],
    )


def _validate(x: torch.Tensor):
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("fused NVFP4 expects 2D BF16 input")
    m, n = x.shape
    if m % 128 or n % 128:
        raise ValueError("fused NVFP4 requires M,N multiples of 128")
    if not x.is_contiguous():
        raise ValueError("fused NVFP4 expects contiguous row-major input")
    if x.device.type != "cuda":
        raise ValueError("fused NVFP4 requires CUDA")


@torch.library.custom_op("nanochat::sm121_fused_nvfp4_quant", mutates_args=())
def fused_quantize(x: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return row_codes, row_sf, col_codes, col_sf, fp32 scalar amax."""
    _validate(x)
    return tuple(_extension().quantize(x))


@fused_quantize.register_fake
def _fake_fused_quantize(x: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("fused NVFP4 expects 2D BF16 input")
    m, n = x.shape
    if m % 128 or n % 128:
        raise ValueError("fused NVFP4 requires M,N multiples of 128")
    if not x.is_contiguous() or x.device.type != "cuda":
        raise ValueError("fused NVFP4 requires contiguous CUDA input")
    return (
        x.new_empty((m, n // 2), dtype=torch.uint8),
        x.new_empty((m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn),
        x.new_empty((n, m // 2), dtype=torch.uint8),
        x.new_empty((n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn),
        x.new_empty((), dtype=torch.float32),
    )


def prewarm() -> None:
    """Compile/loading explicitly before CUDA Graph capture or torch.compile."""
    if not torch.cuda.is_available():
        raise RuntimeError("sm121 fused NVFP4 needs CUDA")
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("sm121 fused NVFP4 requires GB10 compute capability 12.1")
    _extension()
    # Run once: ensure CUDA module load and dynamic dispatch happen pre-capture.
    fused_quantize(torch.zeros((128, 128), device="cuda", dtype=torch.bfloat16))


def enabled() -> bool:
    return os.getenv("NANOCHAT_NVFP4_FUSED_QUANT", "0") == "1"


def torchao_plain_reference(x: torch.Tensor):
    """Reference including the (unfused) torchao global maximum."""
    import torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton as q2
    amax = torch.linalg.vector_norm(x, ord=float("inf"))
    return (*q2.triton_weight_quantize_2d(x, amax), amax)
