"""RMSNorm -> torchao-layout NVFP4 without a materialized BF16 norm.

The quant-only form is used by fused_norm_linear; it does not return BF16
activations. The optional BF16 output is available for residual/multi-consumer
paths and has an analytical RMSNorm backward. Input is (M,K) BF16, both dims
multiples of 128. Uses the same cooperative reduction/quantization CUDA kernel
as fused_quant, including a device-wide maximum of BF16-rounded norm results.
"""
from __future__ import annotations
import os
import torch
from .fused_quant import _extension, _validate, fused_quantize


@torch.library.custom_op("nanochat::sm121_rmsnorm_nvfp4", mutates_args=())
def fused_rmsnorm_quant(
    x: torch.Tensor, eps: float, return_bf16: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate(x)
    return tuple(_extension().rmsnorm_quant(x, eps, return_bf16))


@fused_rmsnorm_quant.register_fake
def _fake(x: torch.Tensor, eps: float, return_bf16: bool = False) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    _validate_shape(x)
    m, n = x.shape
    return (
        x.new_empty((m, n // 2), dtype=torch.uint8),
        x.new_empty((m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn),
        x.new_empty((n, m // 2), dtype=torch.uint8),
        x.new_empty((n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn),
        x.new_empty((), dtype=torch.float32),
        x.new_empty((m, n) if return_bf16 else (0,), dtype=x.dtype),
    )


def _validate_shape(x):
    if (x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[0] % 128
            or x.shape[1] % 128 or not x.is_contiguous()
            or x.device.type != "cuda"):
        raise ValueError("needs contiguous CUDA 2D BF16, dimensions multiples of 128")


def _setup(ctx, inputs, output):
    x, eps, return_bf16 = inputs
    ctx.save_for_backward(x)
    ctx.eps = eps
    ctx.return_bf16 = return_bf16


def rmsnorm_backward(x, grad_y, eps):
    """Analytic RMSNorm gradient computed in FP32, then rounded to BF16."""
    xf = x.float()
    g = grad_y.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (rstd * (g - xf * rstd.square() *
                    (g * xf).mean(dim=-1, keepdim=True))).to(x.dtype)


def _backward(ctx, *grads):
    if not ctx.return_bf16:
        raise RuntimeError(
            "Quant-only RMSNorm has no autograd path by itself. "
            "Use fused_norm_linear to propagate gradients through packed codes."
        )
    x, = ctx.saved_tensors
    g = grads[-1]
    return (rmsnorm_backward(x, g, ctx.eps) if g is not None else None), None, None


fused_rmsnorm_quant.register_autograd(_backward, setup_context=_setup)


@torch._dynamo.allow_in_graph
class _RMSNormNVFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master_weight, shadow, eps):
        from .fused_mlp import _scaled_mm, _weight_quant
        import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nv
        src_shape = x.shape
        orig = x.reshape(-1, x.shape[-1]).contiguous()
        row, rsf, col, csf, amax, _ = fused_rmsnorm_quant(orig, eps, False)
        w = shadow if shadow.numel() else master_weight.to(torch.bfloat16)
        wr, wsf, wgs, wc, wcsf, wamax = _weight_quant(w)
        output = _scaled_mm(row, rsf, amax, wr, wsf, wgs, b_is_tensor_scale=True).reshape(
            *src_shape[:-1], master_weight.shape[0]
        )
        ctx.save_for_backward(x, col, csf, amax, wc, wcsf, wamax)
        ctx.eps = eps
        ctx.master_dtype = master_weight.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from .fused_mlp import _scaled_mm
        x, col, csf, amax, wc, wcsf, wamax = ctx.saved_tensors
        dy = grad_output.contiguous().reshape(-1, grad_output.shape[-1])
        dy_row, dy_rsf, dy_col, dy_csf, dy_amax = fused_quantize(dy)
        # d(norm(x)) = dy @ W, then apply the RMSNorm Jacobian.
        dnorm = _scaled_mm(dy_row, dy_rsf, dy_amax, wc, wcsf, wamax)
        dx = rmsnorm_backward(x, dnorm.reshape_as(x), ctx.eps)
        dw = _scaled_mm(dy_col, dy_csf, dy_amax, col, csf, amax)
        return dx, dw.to(ctx.master_dtype), None, None


def fused_norm_linear(x, linear):
    """One-step RMSNorm -> NVFP4 c_fc for training, no BF16 norm stored."""
    if x.dtype != torch.bfloat16:
        raise ValueError("fused_norm_linear expects BF16 activations")
    if linear.bias is not None or linear.process_group is not None:
        raise ValueError("only unsharded bias-free NVFP4Linear supported")
    eps = torch.finfo(x.dtype).eps  # matches torch.nn.functional.rms_norm default
    shadow = getattr(linear, "_nanochat_bf16_weight", None)
    if shadow is None:
        shadow = x.new_empty((0,))
    return _RMSNormNVFP4Linear.apply(x, linear.weight, shadow, float(eps))


def enabled():
    return os.getenv("NANOCHAT_NVFP4_FUSED_RMSNORM", "0") == "1"
