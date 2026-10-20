#!/usr/bin/env python3
"""Reproducible sm121a fused GEMM comparison (no prerecorded or guessed timings).

Run on a GB10 CUDA 13 / torch 2.14 / torchao 0.19 installation:
  NANOCHAT_FUSED_GEMM_ENABLE=1 python bench_fused_gemm.py --warmup 3 --iters 10

Reference: torchao 2D quantizer -> torch.nn.functional.scaled_mm, whose
torch 2.14 implementation dispatches to torch._scaled_mm_v2. This is the
supported blockwise+tensorwise equivalent of the lower-level _scaled_mm.
Static input scale error is reported against the exact-amax fused result.
The FP4 output's error is always computed AFTER dequantization.

"Effective GB/s" is the logical compulsory input/output traffic divided
by measured time; it is NOT measured DRAM traffic. The CTA rereads A for
each N tile and B for each M tile, so measured traffic may be much higher.
The "saved bytes" column counts only the removed packed intermediate
materialization (write + subsequent read), and does NOT imply equal net
DRAM savings or a throughput improvement.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics

import torch

from nanochat.kernels.fused_gemm import PackedNVFP4, dequantize, fused_gemm


SHAPES = (
    (32768, 1536, 4608),
    (32768, 1536, 6144),
    (32768, 6144, 1536),
    (32768, 32768, 1536),
)


def _clock(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    results = []
    for _ in range(iters):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        y = fn()
        stop.record()
        stop.synchronize()
        results.append(start.elapsed_time(stop))
        del y
    return statistics.median(results)


def _baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Separate torchao quantize + scaled_mm, including global amax kernels."""
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

    amax_a = torch.linalg.vector_norm(a, ord=float("inf"))
    amax_b = torch.linalg.vector_norm(b, ord=float("inf"))
    ac, asf, _, _ = triton_weight_quantize_2d(a, amax_a)
    bc, bsf, _, _ = triton_weight_quantize_2d(b, amax_b)
    F = torch.nn.functional
    # In this target stack, scaled_mm is the supported dispatcher for
    # _scaled_mm_v2 with mixed blockwise(1x16) + tensorwise scaling.
    # No substitute BF16 matmul is silently used if this path fails.
    return F.scaled_mm(
        ac.view(torch.float4_e2m1fn_x2),
        bc.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[asf.flatten(), per_tensor_amax_to_scale(amax_a)],
        scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        scale_b=[bsf.flatten(), per_tensor_amax_to_scale(amax_b)],
        scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        output_dtype=torch.bfloat16,
    )


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    err = (actual.float()-reference.float()).abs().flatten()
    ref = reference.float().square().mean().sqrt().clamp_min(1e-8)
    return {
        "nrmse": float((err.square().mean().sqrt()/ref).item()),
        "p999_abs_over_ref_rms": float((torch.quantile(err,.999)/ref).item()),
        "max_abs_over_ref_rms": float((err.max()/ref).item()),
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Must run on GB10; no invented measurements")
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("sm_121a GB10 required")
    if os.environ.get("NANOCHAT_FUSED_GEMM_ENABLE") != "1":
        raise RuntimeError("NANOCHAT_FUSED_GEMM_ENABLE=1 required")
    from importlib.metadata import version
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(json.dumps({
        "machine": platform.node(), "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(), "torch": torch.__version__,
        "torchao": version("torchao"), "cuda": torch.version.cuda,
        "mode": args.scale_mode, "static_amax_factor": args.static_factor,
        "warmup": args.warmup, "iterations": args.iters, "seed": args.seed,
        "baseline": "torchao 2D quantize + F.scaled_mm (_scaled_mm_v2)",
    }, sort_keys=True))
    print("M,K,N | fused_ms | baseline_ms | fused_TFLOPS | baseline_TFLOPS | "
          "effective_GBps | intermediate_bytes_saved_GiB | "
          "static_vs_exact_nrmse | fp4_out_dequant_nrmse")

    for m,k,n in SHAPES:
        a = torch.randn((m,k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n,k), device="cuda", dtype=torch.bfloat16)
        # Scalar tensor inputs avoid timing Python tensor allocations for the
        # static path; global maxima are outside the timed loop by definition.
        sa=torch.linalg.vector_norm(a,ord=float("inf")).float().reshape(1)
        sb=torch.linalg.vector_norm(b,ord=float("inf")).float().reshape(1)
        if args.scale_mode=="static":
            sa=sa*args.static_factor
            sb=sb*args.static_factor
        kw=(dict(scale_mode="static",static_amax_a=sa,static_amax_b=sb)
            if args.scale_mode=="static" else dict(scale_mode="exact"))

        ms=_clock(lambda:fused_gemm(a,b,**kw),warmup=args.warmup,iters=args.iters)
        baseline_ms=None
        baseline_error=None
        try:
            baseline_ms=_clock(lambda:_baseline(a,b),
                               warmup=args.warmup,iters=args.iters)
        except Exception as exc:
            baseline_error=type(exc).__name__+": "+str(exc)
        ops=2*m*n*k
        # Lower bound of compulsory data traffic: A and B BF16 read once,
        # plus BF16 output written once. Not total CTA or hardware traffic.
        bytes_min=2*(m*k+n*k+m*n)
        # Baseline quantizer materializes codes (4 bits/value) and scales
        # (1 byte / 16 values), then scaled_mm reads them. No weight reuse
        # assumption is made; this is a *logical* intermediate byte count.
        removed_intermediates=2*(m*k+n*k)*(0.5+1/16)
        # Accuracy sample uses <=256 rows to avoid very large quantiles.
        am=a[:args.accuracy_m].contiguous()
        if am.shape[0]%64:
            raise ValueError("--accuracy-m must be divisible by 64")
        ex=fused_gemm(am,b,scale_mode="exact")
        ast=torch.linalg.vector_norm(am,ord=float("inf")).float().reshape(1)
        bst=torch.linalg.vector_norm(b,ord=float("inf")).float().reshape(1)
        ap=fused_gemm(am,b,scale_mode="static",
                      static_amax_a=ast*args.static_factor,
                      static_amax_b=bst*args.static_factor)
        em=_metrics(ap,ex)
        # Compare dequantized output (not uint8 code arithmetic!) against
        # exact-mode BF16 result. Use static output amax 1.25x exact measured.
        omax=torch.linalg.vector_norm(ex,ord=float("inf")).float().reshape(1)
        fp=fused_gemm(am,b,scale_mode="static",
                      static_amax_a=ast*args.static_factor,
                      static_amax_b=bst*args.static_factor,
                      out_fp4=True,out_static_amax=omax*args.static_factor)
        assert isinstance(fp,PackedNVFP4)
        ep=_metrics(dequantize(fp),ex)
        row={
            "shape": [m,k,n], "fused_ms":round(ms,4),
            "baseline_ms":None if baseline_ms is None else round(baseline_ms,4),
            "fused_tflops":round(ops/(ms*1e-3)/1e12,3),
            "baseline_tflops":None if baseline_ms is None else round(ops/(baseline_ms*1e-3)/1e12,3),
            "effective_gbps_lower_bound":round(bytes_min/(ms*1e-3)/1e9,3),
            "baseline_effective_gbps_lower_bound":None if baseline_ms is None else round(bytes_min/(baseline_ms*1e-3)/1e9,3),
            "avoided_packed_intermediate_GiB":round(removed_intermediates/2**30,3),
            "static_vs_exact":em, "fp4_output_dequant_vs_exact":ep,
            "baseline_error":baseline_error,
        }
        print(json.dumps(row,sort_keys=True),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--iters",type=int,default=10)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--scale-mode",choices=("exact","static"),default="static")
    p.add_argument("--static-factor",type=float,default=1.25,
                   help="static global amax / observed amax for error comparison")
    p.add_argument("--accuracy-m",type=int,default=256)
    run(p.parse_args())
