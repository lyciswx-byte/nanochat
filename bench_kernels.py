#!/usr/bin/env python3
"""GB10 (sm_121a) reproducible fused-NVFP4 activation quantizer benchmark.

Usage (NGC 26.09; CUDA 13; torch 2.14a0; torchao 0.19):
  python bench_kernels.py --warmup 3 --iters 10
  python bench_kernels.py --check-only
  NANOCHAT_NVFP4_FUSED_QUANT=1 python bench_kernels.py --autograd-smoke

CUDA event times are per activation, not extrapolated model tok/s.
All MB/GB units in traffic calculations are decimal.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import platform
from collections import OrderedDict

import torch
from nanochat.kernels.fused_quant import fused_quantize, prewarm, torchao_plain_reference
from nanochat.kernels.fused_rmsnorm import fused_rmsnorm_quant
from nanochat.kernels.fused_mlp import fused_relu2_quant
import torch.nn.functional as F


GEMMS = [
    # M, K, N: forward x = (M,K), backward dy = (M,N)
    (32768, 1536, 4608),
    (32768, 1536, 6144),
    (32768, 6144, 1536),
    (32768, 32768, 1536),
]


def dequant(codes, sf, amax):
    """Unsizzle torchao's 32x16 scale atom; decode E2M1 packed nibbles."""
    rows, packed = codes.shape
    cols = packed * 2
    sf_2d = sf.float().view(rows // 128, cols // 64, 32, 4, 4)
    sf_2d = sf_2d.permute(0, 3, 2, 1, 4).contiguous().view(rows, cols // 16)
    # Decoding in chunks keeps the reference from allocating huge FP32 scratch
    # for the full output (the 32768x32768 input is ~2 GiB in BF16 alone).
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nv
    gs = nv.per_tensor_amax_to_scale(amax).float()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=codes.device)
    low = codes & 15
    high = (codes >> 4) & 15
    code = torch.stack((low, high), dim=-1).reshape(rows, cols)
    signed = torch.where((code & 8) != 0, -lut[(code & 7).long()],
                         lut[(code & 7).long()])
    return signed * sf_2d.repeat_interleave(16, dim=1) * gs


def correctness(shape=(128, 256), seed=20261020, compare_dequant=True):
    torch.manual_seed(seed)
    x = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.42).bfloat16()
    # include a zero 16x16 tile, saturation, and sign extremes
    x[:16, :16] = 0
    x[16, 16] = 17
    x[-1, -1] = -21
    with torch.no_grad():
        ref = torchao_plain_reference(x)
        got = fused_quantize(x)
    names = ["row_codes", "row_sf", "col_codes", "col_sf", "amax"]
    issues = {}
    for name, a, b in zip(names, ref, got):
        same = bool(torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a,
                                b.view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b))
        issues[name] = {"bitwise_equal": same}
    if compare_dequant:
        for idx, label, source in ((0, "row", x), (2, "col", x.t())):
            dq_ref = dequant(ref[idx], ref[idx + 1], ref[4])
            dq_new = dequant(got[idx], got[idx + 1], got[4])
            err = (dq_ref - dq_new).abs()
            tol = (dq_ref - source.float()).abs()
            issues[label]["max_reference_delta"] = float(err.max())
            issues[label]["ref_mae_to_bf16"] = float(tol.mean())
            issues[label]["new_mae_to_bf16"] = float((dq_new - source.float()).abs().mean())
    ok = all(v["bitwise_equal"] for v in issues.values())
    return {"shape": shape, "pass": ok, "checks": issues}


def time_it(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def measure(x, args):
    baseline = lambda: torchao_plain_reference(x)
    fused = lambda: fused_quantize(x)
    m, k = x.shape
    # Input reads = 2 passes of BF16, two packed-output orientations,
    # two arrays of one FP8 SF per logical 16-element vector.
    bytes_min = x.numel() * (2 * 2 + 2 * .5 + 2 * (1 / 16))
    ref_ms = time_it(baseline, args.iters, args.warmup)
    fused_ms = time_it(fused, args.iters, args.warmup)
    return {
        "M": m, "features": k, "numel": x.numel(),
        "min_io_GB": round(bytes_min / 1e9, 3),
        "torchao_ms": round(ref_ms, 3),
        "fused_ms": round(fused_ms, 3),
        "speedup": round(ref_ms / fused_ms, 3),
        "torchao_effective_GBs": round(bytes_min / (ref_ms * 1e6), 1),
        "fused_effective_GBs": round(bytes_min / (fused_ms * 1e6), 1),
    }



def transformed_correctness(shape=(128, 256), seed=20261020):
    torch.manual_seed(seed)
    x = (torch.randn(shape, device="cuda") * 0.6).bfloat16()
    eps = float(torch.finfo(x.dtype).eps)
    paths = [
        ("rmsnorm", lambda: F.rms_norm(x, (x.shape[-1],), eps=eps),
         lambda: fused_rmsnorm_quant(x, eps, True)),
        ("relu2", lambda: F.relu(x).square(),
         lambda: fused_relu2_quant(x, True)),
    ]
    res = []
    for name, transform, fused in paths:
        hp = transform()
        reference = torchao_plain_reference(hp)
        result = fused()
        codes_equal = bool(torch.equal(reference[0], result[0]))
        sf_equal = bool(torch.equal(reference[1].view(torch.uint8),
                                    result[1].view(torch.uint8)))
        hp_delta = float((hp.float() - result[5].float()).abs().max())
        # A new scale can move FP4 choices near tie boundaries. Report
        # dequantized delta, not only packed-byte equality.
        dq_ref = dequant(reference[0], reference[1], reference[4])
        dq_new = dequant(result[0], result[1], result[4])
        dequant_mae = float((dq_ref - dq_new).abs().mean())
        res.append({
            "path": name, "shape": shape, "codes_equal": codes_equal,
            "scales_equal": sf_equal, "hp_max_delta": hp_delta,
            "dequant_mae_delta": dequant_mae,
        })
    return res


def measure_transform(x, args, name):
    eps = float(torch.finfo(x.dtype).eps)
    if name == "rmsnorm":
        baseline = lambda: torchao_plain_reference(
            F.rms_norm(x, (x.shape[-1],), eps=eps))
        fused = lambda: fused_rmsnorm_quant(x, eps, False)
    elif name == "relu2":
        baseline = lambda: torchao_plain_reference(F.relu(x).square())
        fused = lambda: fused_relu2_quant(x, False)
    else:
        raise ValueError(name)
    baseline_ms = time_it(baseline, args.iters, args.warmup)
    fused_ms = time_it(fused, args.iters, args.warmup)
    n = x.numel()
    # Baseline: 1 read + 1 write for BF16 transform, then 2 reads for
    # torchao amax/quant, 1B packed codes and 1/8B scale bytes per input.
    baseline_GB = n * (2 + 2 + 2 * 2 + 1 + 1 / 8) / 1e9
    fused_GB = n * (2 * 2 + 1 + 1 / 8) / 1e9
    return {
        "M": x.shape[0], "features": x.shape[1],
        "torchao_ms": round(baseline_ms, 3), "fused_ms": round(fused_ms, 3),
        "speedup": round(baseline_ms / fused_ms, 3),
        "torchao_min_io_GB": round(baseline_GB, 3),
        "fused_min_io_GB": round(fused_GB, 3),
        "saved_activation_GB": round(2 * n / 1e9, 3),
        "torchao_effective_GBs": round(baseline_GB / (baseline_ms / 1000), 1),
        "fused_effective_GBs": round(fused_GB / (fused_ms / 1000), 1),
    }


def autograd_smoke():
    from nanochat.nvfp4 import convert_to_nvfp4_training
    import torch.nn as nn
    os.environ["NANOCHAT_NVFP4_FUSED_QUANT"] = "1"
    m = nn.Linear(128, 128, bias=False, device="cuda", dtype=torch.bfloat16)
    # A standalone nn.Linear is valid; conversion is normally performed
    # on the enclosing model for checkpoint preservation.
    root = nn.Sequential(m)
    convert_to_nvfp4_training(root)
    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.compile(root)(x).float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(root[0].weight.grad).all()
    return "forward/backward torch.compile smoke passed"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--autograd-smoke", action="store_true")
    p.add_argument("--skip-large", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; no simulated benchmark results")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 1):
        raise SystemExit(f"Requires GB10 SM121, got {props.major}.{props.minor}")
    prewarm()
    print(json.dumps({
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "triton": __import__("triton").__version__,
        "torchao": __import__("torchao").__version__,
        "gpu": props.name, "cc": "12.1", "sm_count": props.multi_processor_count,
        "notes": "Measured only on the operator shapes, never claimed as model tok/s",
    }, indent=2))
    for shape in ((128, 128), (128, 256), (256, 128), (256, 256), (1024, 1536)):
        v = correctness(shape)
        print("CORRECTNESS", json.dumps(v))
        if not v["pass"]:
            raise AssertionError(f"torchao layout/code mismatch: {shape}")
    for shape in ((128, 128), (128, 256), (1024, 1536)):
        for result in transformed_correctness(shape):
            print("TRANSFORM_CORRECTNESS", json.dumps(result))
    if args.autograd_smoke:
        print(autograd_smoke())
    if args.check_only:
        return
    for M, K, N in GEMMS:
        if args.skip_large and K >= 32768:
            continue
        for role, features in (("forward_activation", K), ("grad_output", N)):
            x = torch.randn((M, features), device="cuda", dtype=torch.bfloat16)
            r = measure(x, args)
            print("BENCH", json.dumps({"gemm": [M, K, N], "role": role, "path": "amax_quant", **r}))
            for name in ("rmsnorm", "relu2"):
                t = measure_transform(x, args, name)
                print("BENCH", json.dumps({"gemm": [M, K, N], "role": role, "path": name, **t}))
            del x


if __name__ == "__main__":
    main()
