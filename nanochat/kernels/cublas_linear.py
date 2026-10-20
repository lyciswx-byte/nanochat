"""BF16 linear backed by a direct cublasGemmEx call.

PyTorch's matmul reaches ~77-79 TFLOPS at the d24 shapes on GB10 while a
direct cublasGemmEx reaches ~87-89 TFLOPS. The op is opaque to Inductor and
allocates fresh outputs, so it is safe under torch.compile.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc" / "cublas_linear.cu"
    return load(
        name="nanochat_cublas_linear_v1",
        sources=[str(source)],
        verbose=os.environ.get("NANOCHAT_CUBLAS_VERBOSE", "0") == "1",
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"],
        extra_ldflags=["-lcublas"],
    )


@torch.library.custom_op("nanochat::cublas_linear_bf16", mutates_args=())
def cublas_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x @ w.T for 2D contiguous BF16 operands."""
    return _extension().linear(x, w)


@cublas_linear.register_fake
def _cublas_linear_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]))


@torch.library.custom_op("nanochat::cublas_dgrad_bf16", mutates_args=())
def cublas_dgrad(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _extension().dgrad(dy, w)


@cublas_dgrad.register_fake
def _cublas_dgrad_fake(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return dy.new_empty((dy.shape[0], w.shape[1]))


@torch.library.custom_op("nanochat::cublas_wgrad_bf16", mutates_args=())
def cublas_wgrad(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _extension().wgrad(dy, x)


@cublas_wgrad.register_fake
def _cublas_wgrad_fake(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((dy.shape[1], x.shape[1]))


def _setup(ctx, inputs, output):
    x, w = inputs
    ctx.save_for_backward(x, w)


def _backward(ctx, grad):
    x, w = ctx.saved_tensors
    grad = grad.contiguous()
    return cublas_dgrad(grad, w), cublas_wgrad(grad, x)


cublas_linear.register_autograd(_backward, setup_context=_setup)


def enabled() -> bool:
    return os.environ.get("NANOCHAT_CUBLAS_LINEAR", "0") == "1"


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Drop-in for F.linear on BF16; falls back for unsupported shapes."""
    if enabled() and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.is_contiguous() and w.is_contiguous() and x2.shape[-1] % 8 == 0:
            return cublas_linear(x2, w).reshape(*shape[:-1], w.shape[0])
    return torch.nn.functional.linear(x, w.to(dtype=x.dtype))
