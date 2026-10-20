"""Opt-in sm_121a NVFP4 quantization experiments (no trainer monkey-patches).

Reference layout: torchao 0.19 quantize_2d_triton.py. Its *2D* scales
reduce 16x16 source tiles (repeated across 16 rows), not independent
per-row 16-vectors. Swizzle: (outer/128, inner/64, 32, 16) E4M3.
The returned packed codes are uint8 with the even element in the low nibble.

The stand-alone quantizer loads each BF16 activation exactly once from HBM.
The RMSNorm variant uses two launches: a row-reduction producing only one
FP32 rstd per row, then a fused residual-add/normalize/dual-quantize pass;
NO full normalized BF16 tensor is materialized. Thus the fused variant
reads the residual inputs twice. The split is necessary for C=6144, where
holding >=16 complete rows for the columnwise 16x16 reduction is impractical.

IMPORTANT AUTOGRAD CONTRACT: packed integer codes and FP8 scales have no
PyTorch gradient path. rstd is differentiable and has an exact VJP; for
quantized GEMM integration, the GEMM's backward must explicitly call
rmsnorm_residual_backward with its d(normalized) tensor. This exposes the
exact unweighted RMSNorm VJP, but does NOT silently claim STE on FP4 bytes.

All GPU operations are opaque torch.library.custom_op with fresh outputs
and metadata-faithful register_fake. No global cache or donated outputs.
"""
from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        _swizzle_scales,
        convert_8xfp32_to_4xfp4_packed,
    )
except ImportError:  # Source import is safe without the GB10 runtime.
    triton = None
    tl = None


def _enabled() -> None:
    if os.getenv("NANOCHAT_TRITON_FUSED", "0") != "1":
        raise RuntimeError("Set NANOCHAT_TRITON_FUSED=1 to enable experimental operators")
    if triton is None:
        raise RuntimeError("Triton and torchao 0.19 NVFP4 helpers are required")


def _check(x: torch.Tensor, amax: torch.Tensor, residual: torch.Tensor | None = None) -> None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("x must be contiguous BF16 [M,K]")
    m, k = x.shape
    if m % 128 or k % 128:
        raise ValueError("Both M and K must be multiples of 128 (torchao swizzle)")
    if x.device.type != "cuda":
        raise ValueError("GB10 CUDA tensors required")
    if amax.shape != () or amax.dtype != torch.float32 or amax.device != x.device:
        raise ValueError("global_amax must be scalar FP32 on the same GPU")
    if residual is not None and (
        residual.shape != x.shape or residual.dtype != x.dtype
        or residual.device != x.device or not residual.is_contiguous()
    ):
        raise ValueError("residual must match x in shape, dtype, device and layout")


def _resolve_amax(x: torch.Tensor, amax: torch.Tensor | float | None) -> torch.Tensor:
    if amax is None:
        raw = os.getenv("NANOCHAT_NVFP4_STATIC_AMAX")
        if raw is None:
            raise ValueError("Pass global_amax or set NANOCHAT_NVFP4_STATIC_AMAX")
        amax = float(raw)
    if isinstance(amax, torch.Tensor):
        return amax
    val = float(amax)
    if not math.isfinite(val) or val < 0:
        raise ValueError("global_amax must be finite and nonnegative")
    return torch.tensor(val, device=x.device, dtype=torch.float32)


def global_decode_scale(global_amax: torch.Tensor) -> torch.Tensor:
    """Torchao's scalar NVFP4 decode factor (to be passed to a GEMM recipe).

    0 amax has torchao's special decode_scale=1, not 0.
    Compute on-device from the SAME caller scalar used by the kernel.
    """
    safe = torch.where(global_amax == 0, 1.0, global_amax)
    candidate = (2688.0 / safe).clamp(max=torch.finfo(torch.float32).max)
    candidate = torch.where(candidate == 0, 1.0, candidate)
    encode = torch.where(global_amax == 0, 1.0, candidate)
    return 1.0 / encode


def _outputs(x: torch.Tensor):
    m, k = x.shape
    return (
        torch.empty((m, k // 2), device=x.device, dtype=torch.uint8),
        torch.empty((m // 128, k // 64, 32, 16), device=x.device, dtype=torch.float8_e4m3fn),
        torch.empty((k, m // 2), device=x.device, dtype=torch.uint8),
        torch.empty((k // 128, m // 64, 32, 16), device=x.device, dtype=torch.float8_e4m3fn),
    )


if tl is not None:
    @triton.jit
    def _tile_quant(a, global_amax, BM: tl.constexpr, BK: tl.constexpr):
        # Exact association/div_rn of torchao's 2D quantizer.
        a4 = tl.reshape(a, (BM // 16, 16, BK // 16, 16))
        max16 = tl.max(tl.max(tl.abs(a4), 3, keep_dims=True), 1, keep_dims=True)
        safe = tl.where(global_amax == 0, 1.0, global_amax)
        numerator = tl.full((), 2688.0, tl.float32)
        one = tl.full((), 1.0, tl.float32)
        enc = tl.minimum(tl.div_rn(numerator, safe), 3.4028234663852886e38)
        enc = tl.where(enc == 0, 1.0, enc)
        enc = tl.where(global_amax == 0, 1.0, enc)
        dec = tl.div_rn(one, enc)
        # No positive floor for E4M3 block scales (zero blocks remain zero).
        pv = tl.minimum(max16.to(tl.float32) * (enc * (1.0 / 6.0)), 448.0)
        e4 = pv.to(tl.float8e4nv)
        denom = e4.to(tl.float32) * dec
        inv = tl.minimum(tl.div_rn(tl.full(denom.shape, 1.0, tl.float32), denom),
                         3.4028234663852886e38)
        scaled = tl.clamp(a4.to(tl.float32) * inv, -6.0, 6.0)
        return tl.reshape(e4, (BM // 16, BK // 16)), tl.reshape(scaled, (BM, BK))

    @triton.jit
    def _emit_quant(a, amax, codes, sf, t_codes, t_sf,
                    pm, pk, M: tl.constexpr, K: tl.constexpr):
        # Full 128x128 tile: each CTA exclusively owns two 32x16 scale atoms
        # in BOTH orientations. Non-TMA load + tl.trans stages on chip.
        sc, scaled = _tile_quant(a, amax, 128, 128)
        pairs = tl.reshape(scaled, (128, 64, 2)).split()
        packed = convert_8xfp32_to_4xfp4_packed(pairs)
        r = pm * 128 + tl.arange(0, 128)
        c = pk * 64 + tl.arange(0, 64)
        tl.store(codes + r[:, None] * (K // 2) + c[None, :], packed)
        expanded = tl.broadcast_to(tl.expand_dims(sc, 1), (8, 16, 8))
        expanded = tl.reshape(expanded, (128, 8))
        blocked = _swizzle_scales(expanded, 128, 128)
        b = tl.reshape(blocked, (2, 512))
        si = tl.arange(0, 2)
        sj = tl.arange(0, 512)
        offset = (pm * (K // 64) + pk * 2 + si[:, None]) * 512 + sj[None, :]
        tl.store(sf + offset, b)

        a_t = tl.trans(a)
        st, scaled_t = _tile_quant(a_t, amax, 128, 128)
        pairs_t = tl.reshape(scaled_t, (128, 64, 2)).split()
        packed_t = convert_8xfp32_to_4xfp4_packed(pairs_t)
        rt = pk * 128 + tl.arange(0, 128)
        ct = pm * 64 + tl.arange(0, 64)
        tl.store(t_codes + rt[:, None] * (M // 2) + ct[None, :], packed_t)
        expanded_t = tl.reshape(
            tl.broadcast_to(tl.expand_dims(st, 1), (8, 16, 8)), (128, 8)
        )
        blocked_t = tl.reshape(_swizzle_scales(expanded_t, 128, 128), (2, 512))
        offset_t = (pk * (M // 64) + pm * 2 + si[:, None]) * 512 + sj[None, :]
        tl.store(t_sf + offset_t, blocked_t)

    @triton.jit
    def _dual_quant_kernel(
        X, Residual, Rstd, Amax, Codes, Sf, TCodes, TSf,
        M: tl.constexpr, K: tl.constexpr, FUSED: tl.constexpr,
    ):
        pm = tl.program_id(0)
        pk = tl.program_id(1)
        r = pm * 128 + tl.arange(0, 128)
        k = pk * 128 + tl.arange(0, 128)
        a = tl.load(X + r[:, None] * K + k[None, :])
        if FUSED:
            b = tl.load(Residual + r[:, None] * K + k[None, :])
            z = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
            rms = tl.load(Rstd + r)
            a = (z.to(tl.float32) * rms[:, None]).to(tl.bfloat16)
        _emit_quant(a, tl.load(Amax), Codes, Sf, TCodes, TSf, pm, pk, M, K)

    @triton.jit
    def _rstd_kernel(X, Residual, Rstd, K: tl.constexpr, BC: tl.constexpr,
                     EPS: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, BC)
        a = tl.load(X + row * K + c, c < K, other=0).to(tl.float32)
        b = tl.load(Residual + row * K + c, c < K, other=0).to(tl.float32)
        z = (a + b).to(tl.bfloat16).to(tl.float32)
        normsq = tl.sum(z * z, 0) / K
        rms = tl.rsqrt(normsq + EPS)
        tl.store(Rstd + row, rms)

    @triton.jit
    def _backward_kernel(X, Residual, Dy, Rstd, Dx, Dr,
                         K: tl.constexpr, BC: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, BC)
        mask = c < K
        x = tl.load(X + row * K + c, mask, other=0).to(tl.float32)
        y = tl.load(Residual + row * K + c, mask, other=0).to(tl.float32)
        dy = tl.load(Dy + row * K + c, mask, other=0).to(tl.float32)
        z = (x + y).to(tl.bfloat16).to(tl.float32)
        r = tl.load(Rstd + row)
        dot = tl.sum(z * dy, 0) / K
        grad = r * dy - z * (r * r * r * dot)
        tl.store(Dx + row * K + c, grad, mask)
        tl.store(Dr + row * K + c, grad, mask)

    @triton.jit
    def _rstd_vjp_kernel(X, Residual, Rstd, GradRstd, Dx, Dr,
                         K: tl.constexpr, BC: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, BC)
        mask = c < K
        x = tl.load(X + row * K + c, mask, other=0).to(tl.float32)
        y = tl.load(Residual + row * K + c, mask, other=0).to(tl.float32)
        z = (x + y).to(tl.bfloat16).to(tl.float32)
        r = tl.load(Rstd + row)
        g = tl.load(GradRstd + row)
        grad = -g * z * (r * r * r / K)
        tl.store(Dx + row * K + c, grad, mask)
        tl.store(Dr + row * K + c, grad, mask)


@torch.library.custom_op("nanochat::triton_quantize_2d_static", mutates_args=())
def _quantize_op(x: torch.Tensor, global_amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, global_amax)
    out = _outputs(x)  # Fresh on *every invocation*. No persistent cache.
    m, k = x.shape
    _dual_quant_kernel[(m // 128, k // 128)](
        x, x, global_amax, global_amax, *out, m, k, FUSED=False,
        num_warps=8,
    )
    return out


@_quantize_op.register_fake
def _(x, global_amax):
    return _outputs(x)


def _setup_quant(ctx, inputs, output):
    pass


def _back_quant(ctx, *grad_outputs):
    # FP4 codes and FP8 scales are deliberately non-differentiable. A future
    # fused GEMM backward must supply d(normalized) to the RMSNorm VJP below.
    return None, None


_quantize_op.register_autograd(_back_quant, setup_context=_setup_quant)


@torch.library.custom_op("nanochat::triton_residual_rmsnorm_quantize", mutates_args=())
def _rms_quant_op(x: torch.Tensor, residual: torch.Tensor, global_amax: torch.Tensor,
                  eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, global_amax, residual)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    m, k = x.shape
    out = _outputs(x)
    rstd = torch.empty((m,), device=x.device, dtype=torch.float32)
    _rstd_kernel[(m,)](x, residual, rstd, k, triton.next_power_of_2(k), eps,
                      num_warps=4)
    _dual_quant_kernel[(m // 128, k // 128)](
        x, residual, rstd, global_amax, *out, m, k, FUSED=True,
        num_warps=8,
    )
    return (*out, rstd)


@_rms_quant_op.register_fake
def _(x, residual, global_amax, eps):
    return (*_outputs(x), x.new_empty((x.shape[0],), dtype=torch.float32))


@torch.library.custom_op("nanochat::triton_rmsnorm_backward", mutates_args=())
def _rms_bwd_op(x: torch.Tensor, residual: torch.Tensor, grad_normalized: torch.Tensor,
                rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if grad_normalized.shape != x.shape or grad_normalized.dtype != x.dtype:
        raise ValueError("grad_normalized must match x")
    if rstd.shape != (x.shape[0],) or rstd.dtype != torch.float32:
        raise ValueError("rstd must be float32 [M]")
    if x.device != residual.device or x.device != grad_normalized.device or x.device != rstd.device:
        raise ValueError("all inputs must share device")
    dx = torch.empty_like(x)
    dr = torch.empty_like(residual)
    _backward_kernel[(x.shape[0],)](x, residual, grad_normalized, rstd, dx, dr,
                                   x.shape[1], triton.next_power_of_2(x.shape[1]),
                                   num_warps=4)
    return dx, dr


@_rms_bwd_op.register_fake
def _(x, residual, grad_normalized, rstd):
    return torch.empty_like(x), torch.empty_like(residual)


@torch.library.custom_op("nanochat::triton_rstd_vjp", mutates_args=())
def _rstd_vjp_op(x: torch.Tensor, residual: torch.Tensor, rstd: torch.Tensor,
                 grad_rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if grad_rstd.shape != rstd.shape:
        raise ValueError("grad_rstd must have the same shape as rstd")
    dx, dr = torch.empty_like(x), torch.empty_like(residual)
    _rstd_vjp_kernel[(x.shape[0],)](
        x, residual, rstd, grad_rstd.contiguous(), dx, dr,
        x.shape[1], triton.next_power_of_2(x.shape[1]), num_warps=4,
    )
    return dx, dr


@_rstd_vjp_op.register_fake
def _(x, residual, rstd, grad_rstd):
    return torch.empty_like(x), torch.empty_like(residual)


def _setup_rms(ctx, inputs, output):
    x, residual, _, _ = inputs
    ctx.save_for_backward(x, residual, output[-1])


def _back_rms(ctx, grad_codes, grad_sf, grad_t_codes, grad_t_sf, grad_rstd):
    # Integer packed outputs cannot carry d(normalized) in PyTorch autograd.
    # This correctly differentiates the floating rstd output, if consumed.
    if grad_rstd is None:
        return None, None, None, None
    x, residual, rstd = ctx.saved_tensors
    dx, dr = _rstd_vjp_op(x, residual, rstd, grad_rstd)
    return dx, dr, None, None


_rms_quant_op.register_autograd(_back_rms, setup_context=_setup_rms)


def quantize_2d_static(x: torch.Tensor, global_amax: torch.Tensor | float | None = None):
    """Produce torchao-compatible row/column packed codes and blocked scales.

    For a GEMM also pass global_decode_scale(global_amax) as the scalar recipe.
    Env NANOCHAT_TRITON_FUSED defaults OFF; NANOCHAT_NVFP4_STATIC_AMAX is
    used when global_amax is omitted. Pass a preallocated scalar tensor for
    timing (avoids counting host dispatch and scalar H2D allocation).
    """
    _enabled()
    amax = _resolve_amax(x, global_amax)
    return _quantize_op(x, amax)


def residual_rmsnorm_quantize(x: torch.Tensor, residual: torch.Tensor,
                              global_amax: torch.Tensor | float | None = None,
                              eps: float = torch.finfo(torch.bfloat16).eps):
    """Return (row_codes, row_sf, col_codes, col_sf, rstd); no BF16 normalized output."""
    _enabled()
    return _rms_quant_op(x, residual, _resolve_amax(x, global_amax), float(eps))


def rmsnorm_residual_backward(x: torch.Tensor, residual: torch.Tensor,
                              grad_normalized: torch.Tensor, rstd: torch.Tensor):
    """Explicit VJP supplied by downstream GEMM backward.

    dx = dresidual = r*dy - z*(r**3)*mean(z*dy); z=BF16(x+residual).
    The returned grads are freshly allocated (never aliased/cached).
    """
    _enabled()
    return _rms_bwd_op(x, residual, grad_normalized, rstd)


__all__ = [
    "quantize_2d_static", "residual_rmsnorm_quantize",
    "rmsnorm_residual_backward", "global_decode_scale",
]
