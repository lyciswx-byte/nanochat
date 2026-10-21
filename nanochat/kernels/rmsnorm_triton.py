"""Triton RMSNorm (no learnable weight) with a 4-row-per-CTA backward.

The model's norm is plain ``F.rms_norm(x, (C,))``; this module replaces it
behind ``NANOCHAT_TRITON_RMSNORM=1`` with a shape-specialised kernel pair:

  forward : y = x * rsqrt(mean(x^2) + eps), saves r per row
  backward: dx = r*g - x * r^3 * mean(x*g)

Both process four rows per CTA with a 2048-wide reduction tile (masked for
C=1536) and FP32 accumulation. The operators are opaque and allocate fresh
outputs, so they are safe under torch.compile.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(X, Y, R, C: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * 4 + tl.arange(0, 4)
    cols = tl.arange(0, BLOCK)
    mask = cols[None, :] < C
    x = tl.load(X + rows[:, None] * C + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=1) / C
    r = tl.rsqrt(ms + EPS)
    y = (x * r[:, None]).to(tl.bfloat16)
    tl.store(Y + rows[:, None] * C + cols[None, :], y, mask=mask)
    tl.store(R + rows, r)


@triton.jit
def _bwd(X, G, R, DX, C: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * 4 + tl.arange(0, 4)
    cols = tl.arange(0, BLOCK)
    mask = cols[None, :] < C
    x = tl.load(X + rows[:, None] * C + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + rows[:, None] * C + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + rows)
    s = tl.sum(x * g, axis=1) / C
    dx = r[:, None] * g - x * (r * r * r)[:, None] * s[:, None]
    tl.store(DX + rows[:, None] * C + cols[None, :], dx.to(tl.bfloat16), mask=mask)


def _launch_fwd(x2, r):
    c = x2.shape[-1]
    y = torch.empty_like(x2)
    eps = float(torch.finfo(x2.dtype).eps)
    _fwd[(x2.shape[0] // 4,)](x2, y, r, c, eps, triton.next_power_of_2(c), num_warps=8)
    return y


@torch.library.custom_op("nanochat::triton_rmsnorm_fwd", mutates_args=())
def _fwd_op(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    c = x.shape[-1]
    x2 = x.reshape(-1, c)
    r = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
    return _launch_fwd(x2, r).reshape(x.shape), r


@_fwd_op.register_fake
def _fwd_fake(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.numel() // x.shape[-1]
    return torch.empty_like(x), torch.empty((rows,), dtype=torch.float32, device=x.device)


@torch.library.custom_op("nanochat::triton_rmsnorm_bwd", mutates_args=())
def _bwd_op(x: torch.Tensor, g: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    c = x.shape[-1]
    x2 = x.reshape(-1, c)
    g2 = g.reshape(-1, c).contiguous()
    dx = torch.empty_like(x2)
    _bwd[(x2.shape[0] // 4,)](x2, g2, r, dx, c, triton.next_power_of_2(c), num_warps=8)
    return dx.reshape(x.shape)


@_bwd_op.register_fake
def _bwd_fake(x: torch.Tensor, g: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _setup(ctx, inputs, output):
    x, eps = inputs
    ctx.save_for_backward(x, output[1])


def _backward(ctx, grad_y, grad_r):
    x, r = ctx.saved_tensors
    return _bwd_op(x, grad_y.contiguous(), r), None


_fwd_op.register_autograd(_backward, setup_context=_setup)


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    if os.environ.get("NANOCHAT_TRITON_RMSNORM", "0") != "1":
        return torch.nn.functional.rms_norm(x, (x.shape[-1],))
    if x.dim() != 3 or not x.is_contiguous() or x.shape[-1] % 4 or x.shape[-1] > 2048 or x.shape[0] % 4:
        return torch.nn.functional.rms_norm(x, (x.shape[-1],))
    eps = float(torch.finfo(x.dtype).eps)
    return _fwd_op(x, eps)[0]
