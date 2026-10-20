"""Fused ReLU^2 -> packed NVFP4 c_proj training path on SM121a.

The fc1 GEMM still writes its BF16 accumulator output: torchao's CUTLASS
epilogue is unchanged. This path eliminates the *additional* 402 MB BF16
ReLU^2 activation by quantizing the epilogue transform directly into row/col
packed codes. Forward and backward use the same existing SM121 NVFP4 GEMMs.
"""
from __future__ import annotations
import os
import torch
from .fused_quant import _extension, _validate, fused_quantize


@torch.library.custom_op("nanochat::sm121_relu2_nvfp4", mutates_args=())
def fused_relu2_quant(
    x: torch.Tensor, return_bf16: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate(x)
    return tuple(_extension().relu2_quant(x, return_bf16))


@fused_relu2_quant.register_fake
def _fake(x: torch.Tensor, return_bf16: bool = False) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[0] % 128 or x.shape[1] % 128:
        raise ValueError("requires M,N %128==0 contiguous 2D BF16")
    m, n = x.shape
    return (
        x.new_empty((m, n // 2), dtype=torch.uint8),
        x.new_empty((m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn),
        x.new_empty((n, m // 2), dtype=torch.uint8),
        x.new_empty((n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn),
        x.new_empty((), dtype=torch.float32),
        x.new_empty((m, n) if return_bf16 else (0,), dtype=x.dtype),
    )


def _setup(ctx, inputs, output):
    x, return_bf16 = inputs
    ctx.save_for_backward(x)
    ctx.return_bf16 = return_bf16


def _backward(ctx, *grads):
    if not ctx.return_bf16:
        raise RuntimeError("Quant-only ReLU^2 requires fused_relu2_linear for backward")
    x, = ctx.saved_tensors
    g = grads[-1]
    return (g.float() * (2.0 * x.float().relu())).to(x.dtype) if g is not None else None, None


fused_relu2_quant.register_autograd(_backward, setup_context=_setup)


def _weight_quant(w):
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nv
    return nv._weight_quantize_2d(w, False)


def _scaled_mm(a, asf, a_amax, b, bsf, b_amax, b_is_tensor_scale=False):
    """A @ B.T with canonical torchao 32x16 block-scales and tensor-scales."""
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nv
    F = nv.F
    return F.scaled_mm(
        a.view(torch.float4_e2m1fn_x2), b.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[asf.flatten(), nv.per_tensor_amax_to_scale(a_amax)],
        scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        scale_b=[bsf.flatten(), b_amax if b_is_tensor_scale else nv.per_tensor_amax_to_scale(b_amax)],
        scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        output_dtype=torch.bfloat16,
    )


class _ReLU2NVFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fc1_output, master_weight, shadow):
        yshape = fc1_output.shape
        a = fc1_output.reshape(-1, yshape[-1]).contiguous()
        row, row_sf, col, col_sf, amax, _ = fused_relu2_quant(a, False)
        w = shadow if shadow.numel() else master_weight.to(torch.bfloat16)
        wr, wsf, wgs, wc, wcsf, wamax = _weight_quant(w)
        # The W tensor scale from _weight_quantize_2d is already decoded (W_gs).
        import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nv
        F = nv.F
        result = F.scaled_mm(
            row.view(torch.float4_e2m1fn_x2), wr.t(),
            scale_a=[row_sf.flatten(), nv.per_tensor_amax_to_scale(amax)],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[wsf.flatten(), wgs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        ).reshape(*yshape[:-1], master_weight.shape[0])
        ctx.save_for_backward(fc1_output, col, col_sf, amax, wc, wcsf, wamax)
        ctx.master_dtype = master_weight.dtype
        return result

    @staticmethod
    def backward(ctx, grad_output):
        fc1_output, col, col_sf, amax, wc, wcsf, wamax = ctx.saved_tensors
        dy = grad_output.contiguous().reshape(-1, grad_output.shape[-1])
        dyr, dyrsf, dyc, dycsf, dyamax = fused_quantize(dy)
        # Derivative of BF16 ReLU^2, once (before fc1 backward).
        grad_act = _scaled_mm(dyr, dyrsf, dyamax, wc, wcsf, wamax)
        grad_fc1 = (grad_act.float().reshape_as(fc1_output) *
                    (2.0 * fc1_output.float().relu())).to(fc1_output.dtype)
        grad_w = _scaled_mm(dyc, dycsf, dyamax, col, col_sf, amax)
        return grad_fc1, grad_w.to(ctx.master_dtype), None


def fused_relu2_linear(fc1_output, proj):
    """ReLU^2 + quantize + c_proj on packed native SM121 FP4, no BF16 act."""
    if proj.bias is not None or proj.process_group is not None:
        raise ValueError("supports only bias-free unsharded NVFP4Linear")
    if fc1_output.dtype != torch.bfloat16:
        raise ValueError("needs BF16 fc1 GEMM output")
    shadow = getattr(proj, "_nanochat_bf16_weight", None)
    if shadow is None:
        shadow = fc1_output.new_empty((0,))
    return _ReLU2NVFP4Linear.apply(fc1_output, proj.weight, shadow)


def enabled():
    return os.getenv("NANOCHAT_NVFP4_FUSED_MLP", "0") == "1"
