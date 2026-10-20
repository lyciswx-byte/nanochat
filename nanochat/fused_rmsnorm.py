"""Experimental exact-amax RMSNorm path for NVFP4.

The RMSNorm Triton forward computes BF16 output and per-row maxima in the
same read. The torchao plain amax hook reduces only the maxima (O(rows),
rather than O(rows * hidden)). Quantization still reads normalized BF16
once and uses the same global max as the reference path.

This is not a single-kernel norm/FP4 pack: global amax must be known before
the tensor-wide NVFP4 scale can be constructed. Opt in separately from CE.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Ephemeral outputs only; do not retain full activation tensors.
_max_by_storage: dict = {}


@triton.jit
def _norm_forward(X, Y, PARTIAL, C: tl.constexpr, EPS: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < C
    a = tl.load(X + row * C + col, valid, other=0).to(tl.float32)
    mean_sq = tl.sum(a * a, axis=0) / C
    inv = tl.rsqrt(mean_sq + EPS)
    b = (a * inv).to(Y.dtype.element_ty)
    tl.store(Y + row * C + col, b, valid)
    local_max = tl.max(tl.where(valid, tl.abs(b.to(tl.float32)), 0.), axis=0)
    tl.store(PARTIAL + row, local_max)


@triton.jit
def _norm_backward(X, DY, DX, C: tl.constexpr, EPS: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < C
    a = tl.load(X + row * C + col, valid, other=0).to(tl.float32)
    dy = tl.load(DY + row * C + col, valid, other=0).to(tl.float32)
    r = tl.rsqrt(tl.sum(a * a, axis=0) / C + EPS)
    dot = tl.sum(a * dy, axis=0)
    grad = r * dy - a * ((r * r * r) * dot / C)
    tl.store(DX + row * C + col, grad, valid)


def _cache_key(x):
    return (x.data_ptr(), x.numel(), x.dtype, x.device)


@torch.library.custom_op("nanochat::fused_rmsnorm_partial_amax", mutates_args=())
def _fused_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    y = torch.empty_like(x)
    n, c = x.numel() // x.shape[-1], x.shape[-1]
    maxima = torch.empty((n,), dtype=torch.float32, device=x.device)
    _norm_forward[(n,)](x, y, maxima, c, eps, triton.next_power_of_2(c))
    if len(_max_by_storage) >= 256:
        _max_by_storage.clear()
    _max_by_storage[_cache_key(y)] = maxima
    return y


@_fused_rmsnorm.register_fake
def _fused_rmsnorm_fake(x: torch.Tensor, eps: float):
    return torch.empty_like(x)


@torch.library.custom_op("nanochat::fused_rmsnorm_backward", mutates_args=())
def _fused_rmsnorm_backward(x: torch.Tensor, dy: torch.Tensor, eps: float):
    dx = torch.empty_like(x)
    n, c = x.numel() // x.shape[-1], x.shape[-1]
    _norm_backward[(n,)](
        x, dy, dx, c, eps, triton.next_power_of_2(c),
    )
    return dx


@_fused_rmsnorm_backward.register_fake
def _fused_rmsnorm_backward_fake(x, dy, eps: float):
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output):
    x, eps = inputs
    ctx.save_for_backward(x)
    ctx.eps = eps


def _backward(ctx, grad_y):
    (x,) = ctx.saved_tensors
    return _fused_rmsnorm_backward(x, grad_y.contiguous(), ctx.eps), None


_fused_rmsnorm.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op("nanochat::rmsnorm_amax", mutates_args=())
def _cached_rmsnorm_amax(x: torch.Tensor) -> torch.Tensor:
    partial = _max_by_storage.pop(_cache_key(x), None)
    if partial is None:
        return x.abs().amax()
    # An exact max of BF16-rounded normalized values, reduced over rows.
    return partial.amax().to(x.dtype)


@_cached_rmsnorm_amax.register_fake
def _cached_rmsnorm_amax_fake(x):
    return x.new_empty(())


def fused_rmsnorm(x):
    if (not x.is_cuda or not x.is_contiguous() or
            x.shape[-1] > 8192 or x.shape[-1] < 32):
        return F.rms_norm(x, (x.shape[-1],))
    eps = torch.finfo(x.dtype).eps
    return _fused_rmsnorm(x, float(eps))


def rmsnorm_or_plain_amax(x):
    return _cached_rmsnorm_amax(x)


def clear_rmsnorm_amax_cache():
    _max_by_storage.clear()
