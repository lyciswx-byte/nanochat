"""Opt-in sm_121a NVFP4 prologue-fused GEMM.

Public API:
    y = fused_gemm(a_bf16, b_bf16, scale_mode="exact")
    y = fused_gemm(a_bf16, b_bf16, scale_mode="static",
                   static_amax_a=..., static_amax_b=...)
    y = fused_gemm(a_bf16, packed_b=PackedNVFP4(...), ...)
    packed_y = fused_gemm(a_bf16, b_bf16, ..., out_fp4=True,
                          out_static_amax=...)

Matrix B is [N,K], not [K,N]. The packed representation is canonical:
uint8 codes [N,K//2] (low nibble = even K), uint8 E4M3FN
scale factors [N,K//16], FP32 global_scale (NOT amax). A and BF16 B
are quantized deterministically within each GEMM CTA.

Exact mode computes tensor-global amax first; it is NOT a single-read
operation. Static mode bypasses the global reductions but can saturate.
This kernel has NO RHT and cannot claim torchao RHT numerical equivalence.
Autograd uses an STE for FP4 quantization: dA=dY@B, dB=dY.T@A.
Native backward fusion is not claimed; the two gradients are separate
opaque custom_ops so Inductor does not trace inside or donate cached storage.

Set NANOCHAT_FUSED_GEMM_ENABLE=1 explicitly. No training integration,
monkeypatches, persistent buffers, parameter changes, or module imports at
normal training startup. Requires nvcc CUDA 13 and sm_121a / CUDA-enabled torch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional, Union

import torch

_LOCK = Lock()
_EXT = None
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class PackedNVFP4:
    codes: torch.Tensor        # uint8 [rows, K // 2], low nibble = even K
    scales: torch.Tensor       # uint8 [rows, K // 16], raw E4M3FN bytes
    global_scale: torch.Tensor # float32 one element, dequantization multiplier

    @property
    def logical_shape(self) -> tuple[int, int]:
        return (self.codes.shape[0], self.codes.shape[1] * 2)


def _enabled() -> bool:
    return os.environ.get("NANOCHAT_FUSED_GEMM_ENABLE", "0") == "1"


def _extension():
    global _EXT
    if not _enabled():
        raise RuntimeError("Set NANOCHAT_FUSED_GEMM_ENABLE=1 to opt in")
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is None:
            if not torch.cuda.is_available():
                raise RuntimeError("sm_121a CUDA GPU is required")
            from torch.utils.cpp_extension import load
            # PyTorch >=2.14 with CUDA 13 supports the architecture-specific
            # "a" target; do not compile a generic sm_121 image by accident.
            old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
            try:
                _EXT = load(
                    name="nanochat_sm121_fused_gemm_v1",
                    sources=[str(Path(__file__).resolve().parent /
                                 "csrc" / "fused_gemm.cu")],
                    extra_cflags=["-O3", "-std=c++17"],
                    extra_cuda_cflags=[
                        "-O3", "-std=c++17", "-lineinfo",
                        "-gencode=arch=compute_121a,code=sm_121a",
                    ],
                    verbose=os.environ.get("NANOCHAT_FUSED_GEMM_BUILD_VERBOSE", "0") == "1",
                )
            finally:
                if old_arch is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
    return _EXT


def _fake_output(a: torch.Tensor, n: int) -> torch.Tensor:
    return a.new_empty((a.shape[0], n), dtype=torch.bfloat16)


@torch.library.custom_op("nanochat::sm121_fused_gemm_bf16", mutates_args=())
def _forward(a: torch.Tensor, b: torch.Tensor,
             amax_a: torch.Tensor, amax_b: torch.Tensor) -> torch.Tensor:
    return _extension().forward_bf16(a, b, amax_a, amax_b)


@_forward.register_fake
def _(a, b, amax_a, amax_b):
    return _fake_output(a, b.shape[0])


@torch.library.custom_op("nanochat::sm121_fused_gemm_packed", mutates_args=())
def _forward_packed(a: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                    amax_a: torch.Tensor, global_b: torch.Tensor) -> torch.Tensor:
    return _extension().forward_packed(a, codes, scales, amax_a, global_b)


@_forward_packed.register_fake
def _(a, codes, scales, amax_a, global_b):
    return _fake_output(a, codes.shape[0])


@torch.library.custom_op("nanochat::sm121_fused_gemm_out_fp4", mutates_args=())
def _forward_fp4(a: torch.Tensor, b: torch.Tensor,
                 amax_a: torch.Tensor, amax_b: torch.Tensor,
                 amax_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _extension().forward_fp4(a, b, amax_a, amax_b, amax_out)


@_forward_fp4.register_fake
def _(a, b, amax_a, amax_b, amax_out):
    m, n = a.shape[0], b.shape[0]
    return (a.new_empty((m, n // 2), dtype=torch.uint8),
            a.new_empty((m, n // 16), dtype=torch.uint8))


@torch.library.custom_op("nanochat::sm121_fused_gemm_packed_out_fp4", mutates_args=())
def _forward_packed_fp4(a: torch.Tensor, codes: torch.Tensor,
                        scales: torch.Tensor, amax_a: torch.Tensor,
                        global_b: torch.Tensor,
                        amax_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _extension().forward_packed_fp4(
        a, codes, scales, amax_a, global_b, amax_out)


@_forward_packed_fp4.register_fake
def _(a, codes, scales, amax_a, global_b, amax_out):
    m, n = a.shape[0], codes.shape[0]
    return (a.new_empty((m, n // 2), dtype=torch.uint8),
            a.new_empty((m, n // 16), dtype=torch.uint8))


@torch.library.custom_op("nanochat::sm121_fused_gemm_grad_a", mutates_args=())
def _grad_a(dy: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Fresh output. STE uses unquantized B by design; not the FP4 derivative.
    return torch.mm(dy, b)


@_grad_a.register_fake
def _(dy, b):
    return dy.new_empty((dy.shape[0], b.shape[1]))


@torch.library.custom_op("nanochat::sm121_fused_gemm_grad_b", mutates_args=())
def _grad_b(dy: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.mm(dy.transpose(0, 1), a)


@_grad_b.register_fake
def _(dy, a):
    return a.new_empty((dy.shape[1], a.shape[1]))


@torch.library.custom_op("nanochat::sm121_fused_gemm_grad_a_packed", mutates_args=())
def _grad_a_packed(dy: torch.Tensor, codes: torch.Tensor,
                   scales: torch.Tensor, global_b: torch.Tensor) -> torch.Tensor:
    b = dequantize(PackedNVFP4(codes, scales, global_b))
    return torch.mm(dy.float(), b).to(dy.dtype)


@_grad_a_packed.register_fake
def _(dy, codes, scales, global_b):
    return dy.new_empty((dy.shape[0], codes.shape[1] * 2))


def _setup_bf16(ctx, inputs, output):
    a, b, _, _ = inputs
    ctx.save_for_backward(a, b)


def _backward_bf16(ctx, dy):
    a, b = ctx.saved_tensors
    return _grad_a(dy.contiguous(), b), _grad_b(dy.contiguous(), a), None, None


_forward.register_autograd(_backward_bf16, setup_context=_setup_bf16)


def _setup_packed(ctx, inputs, output):
    _, codes, scales, _, global_b = inputs
    ctx.save_for_backward(codes, scales, global_b)


def _backward_packed(ctx, dy):
    codes, scales, global_b = ctx.saved_tensors
    return _grad_a_packed(dy.contiguous(), codes, scales, global_b), None, None, None, None


_forward_packed.register_autograd(_backward_packed, setup_context=_setup_packed)


def dequantize(packed: PackedNVFP4) -> torch.Tensor:
    """Reference reconstruction; returns FP32 [rows,K] (not a GEMM kernel)."""
    c, s, g = packed.codes, packed.scales, packed.global_scale
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=c.device)
    lo = c & 15
    hi = (c >> 4) & 15
    value = torch.empty((c.shape[0], c.shape[1]*2), device=c.device,
                        dtype=torch.float32)
    value[:, 0::2] = torch.where(lo >= 8, -lut[(lo & 7).long()],
                                lut[(lo & 7).long()])
    value[:, 1::2] = torch.where(hi >= 8, -lut[(hi & 7).long()],
                                lut[(hi & 7).long()])
    block = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return value * block * g.float()


def _scalar(x: Union[torch.Tensor, float], a: torch.Tensor, label: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.numel() != 1 or x.device != a.device:
            raise ValueError(f"{label} must be a 1-element scalar on A's CUDA device")
        return x.float().reshape(1).contiguous()
    v = float(x)
    if not (v >= 0 and v < float("inf")):
        raise ValueError(f"{label} must be finite and nonnegative")
    return torch.tensor([v], dtype=torch.float32, device=a.device)


def _amax(a: torch.Tensor, scale_mode: str,
          given: Optional[Union[torch.Tensor, float]], label: str) -> torch.Tensor:
    if scale_mode == "exact":
        if given is not None:
            raise ValueError(f"{label}: exact mode must measure the tensor")
        # Reduces before the GEMM: global synchronization *between* kernels.
        return a.float().abs().amax().reshape(1)
    if scale_mode != "static":
        raise ValueError("scale_mode must be 'exact' or 'static'")
    if given is None:
        val = os.environ.get("NANOCHAT_NVFP4_STATIC_AMAX")
        if val is None:
            raise ValueError("static mode requires a caller value or "
                             "NANOCHAT_NVFP4_STATIC_AMAX")
        given = float(val)
    return _scalar(given, a, label)


def _validate(a: torch.Tensor, b: Union[torch.Tensor, PackedNVFP4]) -> None:
    if not _enabled():
        raise RuntimeError("NANOCHAT_FUSED_GEMM_ENABLE=1 required (default OFF)")
    if a.dtype != torch.bfloat16 or a.ndim != 2 or not a.is_cuda or not a.is_contiguous():
        raise ValueError("A must be contiguous CUDA BF16 [M,K]")
    if a.shape[0] % 64 or a.shape[1] % 64:
        raise ValueError("M and K must be divisible by 64")
    if torch.cuda.get_device_capability(a.device) != (12, 1):
        raise RuntimeError("sm_121a is the only supported target")
    if isinstance(b, PackedNVFP4):
        if (b.codes.dtype != torch.uint8 or b.scales.dtype != torch.uint8 or
            b.codes.ndim != 2 or b.scales.ndim != 2 or
            b.codes.shape[1] * 2 != a.shape[1] or
            b.codes.shape[0] != b.scales.shape[0] or
            b.scales.shape[1] != a.shape[1] // 16 or
            b.codes.shape[0] % 64 or
            not b.codes.is_contiguous() or not b.scales.is_contiguous() or
            b.codes.device != a.device or b.scales.device != a.device):
            raise ValueError("packed B must be canonical contiguous CUDA uint8 codes/scales")
        if (b.global_scale.device != a.device or b.global_scale.dtype != torch.float32 or
            b.global_scale.numel() != 1):
            raise ValueError("packed B global_scale must be one CUDA FP32 value")
    else:
        if (b.dtype != torch.bfloat16 or b.ndim != 2 or b.shape[1] != a.shape[1] or
            b.shape[0] % 64 or not b.is_cuda or b.device != a.device or
            not b.is_contiguous()):
            raise ValueError("B must be contiguous CUDA BF16 [N,K], N divisible by 64")


def fused_gemm(
    a: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    *,
    packed_b: Optional[PackedNVFP4] = None,
    scale_mode: str = "exact",
    static_amax_a: Optional[Union[torch.Tensor, float]] = None,
    static_amax_b: Optional[Union[torch.Tensor, float]] = None,
    out_fp4: bool = False,
    out_static_amax: Optional[Union[torch.Tensor, float]] = None,
) -> Union[torch.Tensor, PackedNVFP4]:
    """Y=A@B.T. A is BF16, B BF16 or canonical already-packed NVFP4.

    BF16 B supports STE gradients w.r.t. both A and B through registered
    opaque custom_op backward operators. Packed B supports A gradients only.
    A packed FP4 output has no autograd and needs an independent static global
    output amax; exact output amax is impossible inside a one-pass epilogue.

    In static mode, the supplied *_amax values are *amax*, not global_scale.
    The kernel computes global_scale = amax / (6*448).
    """
    if (b is None) == (packed_b is None):
        raise ValueError("specify exactly one of b or packed_b")
    inp = packed_b if packed_b is not None else b
    assert inp is not None
    _validate(a, inp)
    amax_a = _amax(a, scale_mode, static_amax_a, "A amax")
    if isinstance(inp, PackedNVFP4):
        if static_amax_b is not None:
            raise ValueError("packed B already has a global_scale; don't supply B amax")
        if out_fp4:
            if a.requires_grad:
                raise ValueError("out_fp4 is nondifferentiable; request BF16 for autograd")
            if out_static_amax is None:
                raise ValueError("out_fp4 requires out_static_amax")
            outmax = _scalar(out_static_amax, a, "output amax")
            codes, scales = _forward_packed_fp4(
                a, inp.codes, inp.scales, amax_a, inp.global_scale, outmax)
            return PackedNVFP4(codes, scales, outmax / 2688.0)
        return _forward_packed(a, inp.codes, inp.scales, amax_a, inp.global_scale)
    assert isinstance(inp, torch.Tensor)
    amax_b = _amax(inp, scale_mode, static_amax_b, "B amax")
    if out_fp4:
        if a.requires_grad or inp.requires_grad:
            raise ValueError("out_fp4 is nondifferentiable; request BF16 for autograd")
        if out_static_amax is None:
            raise ValueError("out_fp4 requires out_static_amax (no global barrier)")
        outmax = _scalar(out_static_amax, a, "output amax")
        codes, scales = _forward_fp4(a, inp, amax_a, amax_b, outmax)
        return PackedNVFP4(codes, scales, outmax / 2688.0)
    return _forward(a, inp, amax_a, amax_b)


__all__ = ["PackedNVFP4", "fused_gemm", "dequantize"]
