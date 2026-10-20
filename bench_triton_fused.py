"""Reproducible GB10 NVFP4 correctness, timing and overlap *prototype*.

RUN ON A REAL DGX SPARK (not run by the author):
  NANOCHAT_TRITON_FUSED=1 NANOCHAT_NVFP4_STATIC_AMAX=6.0 \
    python bench_triton_fused.py --mode all --warmup 5 --repeats 20 --seed 20261020

--mode correctness: full d24-sized BF16 tensors versus dynamic torchao
--mode micro: isolated quantizer, torchao quantizer, fused RMSNorm
--mode overlap: real d24 NVFP4Linears and captured *real* model activations
--mode real-step: d24 forward+backward on synthetic tokens (no optimizer)
--mode all: all of the above

IMPORTANT: replay overlap pre-materializes both layer inputs. It is an
independent-work upper bound, NOT a legal same-forward layer-i/i+1 schedule.
The dependency-respecting variant explicitly waits for the preceding GEMM.
Report real-step timing independently; do not add per-op or replay improvements
to an end-to-end result. CUDA events time queued GPU work (not Python setup).
Nothing is extrapolated as a measured speedup.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from pathlib import Path

import torch

from nanochat.kernels.triton_fused import (
    global_decode_scale, quantize_2d_static, residual_rmsnorm_quantize,
)

BASE_SHA = "57d056b4f056b24c3ef33ade40b6430286b087e7"
BRANCH = "chatgpt/sm121-triton-overlap-20261020-0530"
SHAPES = ((32768, 1536), (32768, 6144))
PROFILE = {"microstep_s": 2.18, "gemm_s": 0.54, "nongemm_s": 1.64,
           "activation_quant_s": 0.36, "amax_s": 0.19, "rmsnorm_s": 0.19}
_MAGNITUDES = (0, .5, 1, 1.5, 2, 3, 4, 6)


def dequantize(q: torch.Tensor, sf: torch.Tensor, global_amax: torch.Tensor,
               chunk_rows: int = 128) -> torch.Tensor:
    """Decode exact 32x16 scale atom layout and low/high FP4 nibble order.

    Full output is BF16 to cap peak memory; comparisons below upcast chunkwise.
    No reference to the quantizer implementation is used in this decoder.
    """
    outer, half_inner = q.shape
    inner = half_inner * 2
    scales = sf.view(outer // 128, inner // 64, 32, 4, 4)
    scales = scales.permute(0, 3, 2, 1, 4).contiguous()
    scales = scales.view(outer, inner // 16)
    result = torch.empty((outer, inner), device=q.device, dtype=torch.bfloat16)
    lut = torch.tensor(_MAGNITUDES, device=q.device, dtype=torch.float32)
    global_sf = global_decode_scale(global_amax).float()
    for row in range(0, outer, chunk_rows):
        block = q[row:row + chunk_rows].to(torch.int32)
        code = torch.stack((block & 15, (block >> 4) & 15), dim=-1).flatten(-2)
        mag = lut[(code & 7).long()]
        val = torch.where((code & 8) != 0, -mag, mag)
        decode_sf = scales[row:row + chunk_rows].float().repeat_interleave(16, -1)
        result[row:row + chunk_rows] = (val * decode_sf * global_sf).to(torch.bfloat16)
    return result


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    # FP32 arithmetic and one scalar transfer. Avoid holding extra FP32 copies.
    return float(((a.float() - b.float()).square().mean().sqrt()).item())


def _stats(candidate, reference, x, amax) -> dict:
    out = {}
    for prefix, matrix, idx in (("row", x, (0, 1)), ("column", x.T, (2, 3))):
        a = dequantize(candidate[idx[0]], candidate[idx[1]], amax)
        b = dequantize(reference[idx[0]], reference[idx[1]], x.abs().max().float())
        out[prefix] = {
            "candidate_rmse_vs_input": _rmse(a, matrix),
            "reference_rmse_vs_input": _rmse(b, matrix),
            "candidate_vs_reference_rmse": _rmse(a, b),
            "equal_packed_bytes": bool(torch.equal(candidate[idx[0]], reference[idx[0]])),
            "equal_swizzled_scales": bool(torch.equal(candidate[idx[1]], reference[idx[1]])),
            "zero_scale_rate": float((candidate[idx[1]].float() == 0).float().mean().item()),
        }
        del a, b
    return out


def _amax_tensor(x: torch.Tensor, cli_value: float | None) -> torch.Tensor:
    raw = os.getenv("NANOCHAT_NVFP4_STATIC_AMAX")
    value = cli_value if cli_value is not None else float(raw) if raw else 6.0
    if not (0.0 <= value < float("inf")):
        raise ValueError("static amax must be nonnegative finite")
    return torch.tensor(value, dtype=torch.float32, device=x.device)


def _time_cuda(fn, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return {"median_ms": statistics.median(times), "min_ms": min(times),
            "max_ms": max(times), "repeat_ms": times}


def _torchao_reference(x, dyn_amax):
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )
    return triton_weight_quantize_2d(x, dyn_amax)


def correctness(args, rng) -> list[dict]:
    out = []
    for m, k in SHAPES:
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=rng)
        # A single dynamic-amax prepass for the reference. NEVER in the timed static kernel.
        dynamic = x.abs().max().float()
        static = _amax_tensor(x, args.static_amax)
        ref = _torchao_reference(x, dynamic)
        got = quantize_2d_static(x, static)
        torch.cuda.synchronize()
        entry = {
            "shape": [m, k], "static_amax": float(static.item()),
            "dynamic_amax": float(dynamic.item()),
            "errors": _stats(got, ref, x, static),
        }
        # Bit-exact regression: with the same global_amax, both paths must
        # agree in packed FP4 and E4M3 scale atoms.
        same_amax = quantize_2d_static(x, dynamic)
        entry["exact_dynamic_scale_equivalence"] = all(
            torch.equal(left, right) for left, right in zip(same_amax, ref)
        )
        if not entry["exact_dynamic_scale_equivalence"]:
            raise AssertionError(f"not bit-exact to torchao at {m}x{k}; inspect 2D scales")
        out.append(entry)
        print(json.dumps({"correctness": entry}, sort_keys=True), flush=True)
        del x, ref, got, same_amax
    return out


def micro(args, rng) -> list[dict]:
    out = []
    for m, k in SHAPES:
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=rng)
        residual = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=rng)
        static = _amax_tensor(x, args.static_amax)
        dynamic = x.abs().max().float()  # not timed
        try:
            x_run = _time_cuda(lambda: quantize_2d_static(x, static), args.warmup, args.repeats)
            ref_run = _time_cuda(lambda: _torchao_reference(x, dynamic),
                                 args.warmup, args.repeats)
            fused_run = _time_cuda(
                lambda: residual_rmsnorm_quantize(x, residual, static),
                args.warmup, args.repeats,
            )
            row = {"shape": [m, k], "fused_static_quant": x_run,
                   "dynamic_torchao_quant_excluding_amax": ref_run,
                   "fused_residual_rmsnorm_quant": fused_run}
            out.append(row)
            print(json.dumps({"micro": row}, sort_keys=True), flush=True)
        finally:
            del x, residual
    return out


def build_d24():
    # The real nanochat architecture and real NVFP4Linears, WITHOUT touching
    # base_train.py or gpt.py. No tokenizer dataset/checkpoint required.
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.nvfp4 import convert_to_nvfp4_training, fuse_qkv_projections
    config = GPTConfig(sequence_len=2048, vocab_size=32768,
                       n_layer=24, n_head=12, n_kv_head=12,
                       n_embd=1536, window_pattern="SSSL")
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    fuse_qkv_projections(model)
    count = convert_to_nvfp4_training(model, use_rht=False)
    if count <= 0:
        raise RuntimeError("d24 was not converted to torchao NVFP4Linears")
    return model, count


@torch.no_grad()
def capture_layer_inputs(model, seed: int):
    # Actual per-layer activations from the d24 model, not independent Gaussian
    # tensors. Capturing them in advance makes the replay overlap only an
    # *upper bound*; real layer i+1 input depends on layer i computation.
    seen = {}
    modules = {
        "layer0_fc": model.transformer.h[0].mlp.c_fc,
        "layer1_fc": model.transformer.h[1].mlp.c_fc,
    }
    hooks = []
    for name, module in modules.items():
        def save_input(_module, inp, *, key=name):
            if key not in seen:
                seen[key] = inp[0].detach().reshape(-1, 1536).contiguous().clone()
            # Returning None is essential: a pre-hook return value REPLACES the
            # live model input and would change its 3D shape to a 2D replay tile.
            return None
        hooks.append(module.register_forward_pre_hook(save_input))
    try:
        rng = torch.Generator(device="cuda").manual_seed(seed + 1)
        tokens = torch.randint(0, model.config.vocab_size, (16, 2048),
                               device="cuda", generator=rng)
        _ = model(tokens)  # full d24 forward, real attention and MLPs
        torch.cuda.synchronize()
    finally:
        for h in hooks:
            h.remove()
    assert set(seen) == set(modules), "failed to capture both d24 activations"
    return seen


def _schedule_pair(gemm, quant, quant_stream, dependent: bool):
    """GPU-event makespan, including handoff and both streams' completion."""
    origin = torch.cuda.current_stream()
    begin = torch.cuda.Event(enable_timing=True)
    gemm_done = torch.cuda.Event(enable_timing=False)
    quant_done = torch.cuda.Event(enable_timing=False)
    finish = torch.cuda.Event(enable_timing=True)
    begin.record(origin)
    if dependent:
        # MUST record gemm_done BEFORE asking another stream to wait on it.
        # CUDA waits on never-recorded events are not a real dependency!
        gemm()
        gemm_done.record(origin)
        with torch.cuda.stream(quant_stream):
            quant_stream.wait_event(gemm_done)
            quant()
            quant_done.record(quant_stream)
    else:
        # Both inputs were independently captured earlier; this is upper-bound
        # replay, not a live cross-layer forward pass.
        with torch.cuda.stream(quant_stream):
            quant_stream.wait_event(begin)
            quant()
            quant_done.record(quant_stream)
        gemm()
        gemm_done.record(origin)
    origin.wait_event(quant_done)
    finish.record(origin)
    return finish, begin


def overlap(args):
    model, n_converted = build_d24()
    model.eval()
    saved = capture_layer_inputs(model, args.seed)
    x_i, x_next = saved["layer0_fc"], saved["layer1_fc"]
    amax = _amax_tensor(x_next, args.static_amax)
    gemm = lambda: model.transformer.h[0].mlp.c_fc(x_i)
    quant = lambda: quantize_2d_static(x_next, amax)
    stream = torch.cuda.Stream()
    serial_g = _time_cuda(gemm, args.warmup, args.repeats)
    serial_q = _time_cuda(quant, args.warmup, args.repeats)

    def run_variant(dependent):
        for _ in range(args.warmup):
            end, _ = _schedule_pair(gemm, quant, stream, dependent)
            end.synchronize()
        durations = []
        for _ in range(args.repeats):
            end, start = _schedule_pair(gemm, quant, stream, dependent)
            end.synchronize()
            durations.append(start.elapsed_time(end))
        return {"median_ms": statistics.median(durations), "repeat_ms": durations}

    independent = run_variant(False)
    actual_dependency = run_variant(True)
    g = serial_g["median_ms"]
    q = serial_q["median_ms"]
    hide = max(0.0, min(q, g + q - independent["median_ms"]))
    relative_hide = hide / q if q else 0
    result = {
        "source": "real d24 NVFP4 linear; real captured layer-0 and layer-1 MLP activations",
        "mode": "pre-captured layer-input replay, no optimizer, no backward",
        "converted_linears": n_converted,
        "cuda_events": True,
        "gemm_callable": "NVFP4Linear forward (INCLUDES torchao input/weight quantization)",
        "gemm_forward_ms": serial_g, "static_quant_ms": serial_q,
        "independent_replay_ms": independent,
        "dependency_respecting_ms": actual_dependency,
        "replay_hidden_ms_per_pair": hide,
        "replay_hidden_fraction_of_quant": relative_hide,
        "profiled_nongemm_s": PROFILE["nongemm_s"],
        "optimistic_hidden_activation_quant_s": min(PROFILE["activation_quant_s"] * relative_hide,
                                                     PROFILE["gemm_s"]),
        "caveat": "Layer i+1 activation is causally downstream of GEMM i. Replay gain is "
                  "NOT schedulable on the live forward. Dependent timeline is the "
                  "relevant lower bound without graph restructuring.",
    }
    print(json.dumps({"overlap": result}, sort_keys=True), flush=True)
    return result


def real_step(args):
    model, n = build_d24()
    model.train()
    rng = torch.Generator(device="cuda").manual_seed(args.seed + 99)
    inp = torch.randint(0, model.config.vocab_size, (16, 2048), device="cuda", generator=rng)
    targets = inp.roll(-1, 1)

    def step():
        model.zero_grad(set_to_none=True)
        loss = model(inp, targets)
        loss.backward()
        return loss

    # Default eager benchmark, not assumed equivalent to torch.compile baseline.
    def timed():
        return _time_cuda(step, min(args.warmup, 2), min(args.repeats, 5))

    measurements = timed()
    result = {"mode": "real d24 fwd+bwd, single GB10, synthetic token ids, eager",
              "batch": 16, "seq": 2048, "depth": 24, "n_embd": 1536,
              "converted_nvfp4_linears": n, "timing": measurements,
              "baseline_from_repo_s": PROFILE["microstep_s"],
              "warning": "2.18s baseline used a separately profiled training setup; "
                         "do not equate eager synthetic data with compiled training."}
    print(json.dumps({"real_step": result}, sort_keys=True), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "micro", "overlap", "real-step", "all"],
                    default="correctness")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--static-amax", type=float, default=None)
    ap.add_argument("--json", default=None, help="write metadata and results to JSON")
    args = ap.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        ap.error("warmup must be >=0, repeats must be >=1")
    if os.getenv("NANOCHAT_TRITON_FUSED") != "1":
        ap.error("Set NANOCHAT_TRITON_FUSED=1; all experimental operators default off")
    if not torch.cuda.is_available():
        ap.error("DGX Spark CUDA device required")
    prop = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability()
    if cap != (12, 1):
        ap.error(f"Expected sm_121 GB10, found compute capability {cap}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    import triton
    try:
        import torchao
        torchao_ver = getattr(torchao, "__version__", "unknown")
    except ImportError:
        torchao_ver = "missing"
    config = {
        "repo": "lyciswx-byte/nanochat", "base_sha": BASE_SHA, "branch": BRANCH,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": triton.__version__, "torchao": torchao_ver,
        "python": platform.python_version(), "gpu": prop.name,
        "compute_capability": list(cap), "sm_count": prop.multi_processor_count,
        "seed": args.seed, "shapes": SHAPES,
        "warmup": args.warmup, "repeats": args.repeats,
        "static_amax_arg": args.static_amax,
        "static_amax_env": os.getenv("NANOCHAT_NVFP4_STATIC_AMAX"),
        "profile_from_base_sha_seconds": PROFILE,
        "timing": "CUDA events with end.synchronize; no profiler or host time",
        "environment": {key: val for key, val in os.environ.items()
                        if key.startswith(("NANOCHAT_NVFP4_", "NANOCHAT_TRITON_FUSED",
                                           "CUDA_VISIBLE_DEVICES", "TORCHINDUCTOR_"))},
    }
    results = {}
    if args.mode in {"correctness", "all"}:
        results["correctness"] = correctness(args, rng)
    if args.mode in {"micro", "all"}:
        results["micro"] = micro(args, rng)
    if args.mode in {"overlap", "all"}:
        results["overlap"] = overlap(args)
    if args.mode in {"real-step", "all"}:
        results["real_step"] = real_step(args)
    record = {"metadata": config, "results": results,
              "status": "measured_on_hardware_by_script_runner"}
    if args.json:
        Path(args.json).write_text(json.dumps(record, sort_keys=True, indent=2))
    print(json.dumps({"summary": {"metadata": config, "result_sections": list(results)}},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
