# d24 NVFP4 training throughput on 2x DGX Spark (GB10)

Measured 2026-10-20 on two GB10 Sparks (121 GB unified memory each, 200 Gbps
RoCE stacking cable), NGC PyTorch 26.09 (torch 2.14, CUDA 13), torchao 0.19.

## Configuration

```
--depth=24 --device-batch-size=16 --window-pattern=SSSL
--nvfp4 --nvfp4-fuse-qkv
BF16 master weights, clone-backed weight-quant cache
```

## Measured

| Metric | Value |
|---|---|
| Single GB10, fwd+bwd micro-step (B=16, T=2048) | **2.18 s** |
| Single GB10 peak allocation | 43.9 GiB |
| Dual node, plain quantizer | **~30.0k tok/s** (35.0 s / optimizer step) |
| Dual node, RHT quantizer (`--nvfp4-rht`) | **~25k tok/s** (stable) |
| Host memory, dual node | ~68 GB peak (< 110 GB) |
| d24 NVFP4, plain quantizer stability | NaN after 2–3 optimizer steps |
| d24 NVFP4, RHT stability | finite through 4 steps, loss decreasing |

## Where the 2.18 s goes (torch.profiler, steady state)

| Component | s / micro-step | Share |
|---|---|---|
| NVFP4 GEMMs | 0.54 | 24% |
| Activation quantize (`triton_quantize_2d_weight`, 194 calls) | 0.36 | 16% |
| FlashAttention-2 fwd+bwd | 0.32 | 14% |
| RMSNorm fwd+bwd | 0.19 | 9% |
| global-amax reductions | 0.19 | 9% |
| qkv cat / residual elementwise | 0.07 | 3% |
| embeddings | 0.10 | 5% |
| LM head / CE | 0.03 | 1% |
| other small kernels | ~0.39 | 18% |

## Why 60k tok/s is not reachable with this stack

60k dual-node means 1.09 s per GPU micro-step, i.e. 132 TFLOP/s/GPU of model
FLOPs (63% of the 209 TFLOP/s BF16 peak, 26% of the 500 TFLOP/s NVFP4 peak).

- The GEMMs already run at ~54–68% of NVFP4 peak and take 0.54 s. Even with
  zero non-GEMM work the ceiling would be ~0.54 s.
- The other 1.64 s is memory-bound. The GB10 sustains only ~200 GB/s, and one
  micro-step moves roughly 340 GB because every activation is written by one
  kernel and re-read by the quantizer, the GEMM, the amax reduction and the
  next kernel.
- 60k requires ~110 GB per micro-step, i.e. essentially every activation
  touched once. That needs full fusion: quantize in the GEMM epilogue/prologue
  (CUTLASS block-scaled epilogue), fused MLP (fc1 → relu² → quantize → fc2),
  fused residual + RMSNorm + quantize, and multi-stream overlap. This is a
  kernel-library project, not a configuration change.

## Correctness notes

- The previous weight-quant cache returned persistent packed tensors from an
  opaque custom op. Inductor owns custom-op outputs and donates their storage,
  so the packed weights were silently overwritten: a compiled forward repeated
  with unchanged weights went from loss 10.3976 to 24.81, and training NaNs.
  The cache now returns clones.
- A non-persistent BF16 shadow buffer is baked into the compiled graph by
  Inductor; copying new master values after `optimizer.step()` never reaches
  the forward, so training optimises stale weights and diverges. BF16 master
  weights are used instead.

## Negative results (round 2)

- **Batch 24**: 3266 ms for 24x2048 tokens, exactly the same 15.0k tok/s/GPU as
  B=16 (2179 ms for 16x2048). Per-token cost is flat in batch size, so a larger
  micro-batch does not amortise anything. B=32 needs ~100 GB of GPU allocation
  and would exceed the 110 GB host budget.
- **Quack dense GEMM**: `quack.gemm` fails on sm_121 with
  `cudaErrorNoKernelImageForDevice` (18/18 autotune configs rejected); torch's
  plain BF16 mm reaches 83 TFLOP/s. Quack's block-scaled path runs but is
  slower than torchao's `scaled_mm` here, so it is not a drop-in replacement.
- **CUDA graphs** (`mode="reduce-overhead"`): capture fails with a data-pointer
  error because the weight cache creates tensors outside the capture. Making it
  graph-safe needs the quantized weights expressed as real buffers updated in
  place, which is part of the fused-kernel work.

## Round 3: stability is the second blocker

| Path (d24, B=16, single node, 4 steps) | tok/s/GPU | loss trajectory |
|---|---|---|
| FP8 | 11.2k | 10.3984 -> 10.3988, finite |
| NVFP4 + RHT | 12.6k | 10.3973 -> 10.2986, finite |
| NVFP4 plain | 15.0k | NaN at step 3 |

On the **two-node** setup, NVFP4 + RHT also reaches NaN at step 2
(`dt` 42.3 s -> 38.8 s, loss `10.373` -> `nan`), so RHT does not make the fast
path usable there. FP8 stays finite and is the only stable low-precision path
measured so far, at roughly 22k tok/s dual-node.

So the objective is blocked on two independent, measured conditions:

1. **Throughput.** 60k needs 1.09 s per GPU micro-step; the measured step is
   2.18 s. GEMMs alone are 0.54 s and already run at 54-68% of NVFP4 peak. The
   other 1.64 s is memory-bound and needs roughly a 3x traffic reduction, which
   requires fused quantize+GEMM / fused-MLP kernels that do not exist for
   `sm_121` in torchao, PyTorch or Quack (Quack's dense kernels have no
   `sm_121` image; CUDA-graph capture fails with the weight cache).
2. **Stability.** The 30k tok/s NVFP4 path diverges on d24 (NaN at step 2-3,
   plain or RHT, single or dual node). The stable FP8 path is ~22k tok/s.

## BF16 (BF16 master) — measured ceiling

Two-node, d24, B=16, T=2048, 4 steps, `NANOCHAT_MASTER_BF16=1`, no low precision:

| step | tok/s | loss |
|---|---|---|
| 0 | 19,304 | 10.397177 |
| 1 | 20,246 | 10.369998 |
| 2 | 20,049 | 10.323536 |
| 3 | 20,187 | 10.252126 |

Single-node micro-step 3.14 s (fwd 1.02 + bwd 2.12), 54.5 GB peak.

### Why 30k is outside the roofline

- BF16 matmul peak measured on GB10: **81-83 TFLOPS** (8192^3), **74-78 TFLOPS**
  at the model shapes. Plain `torch.mm` and ATEN-autotuned are identical.
  Inductor's CUTLASS templates abort on sm_121 (`architecture > 107`), and
  max-autotune is disabled below 68 SMs (GB10 has 48).
- Model matmul work is 143.5 TFLOP per micro-step, so GEMM alone is 1.73-1.91 s.
- 30k dual-node = 15k/GPU = 2.18 s per micro-step, leaving 0.29-0.45 s for
  everything else. Non-GEMM measured: attention 0.31 s, ReLU^2 0.24 s,
  RMSNorm 0.20 s, embeddings/CE/add 0.50 s = 1.28 s.
- Even with zero non-GEMM the BF16 step cannot go below ~1.73 s.

Tried and rejected: `mode="reduce-overhead"` (3.148 s), `max-autotune-no-cudagraphs`
(3.210 s, gate disabled), fused QKV for BF16 (3.163 s vs 3.126 s), ATEN GEMM
autotune (77 vs 78 TFLOPS). BF16 ceiling on this stack is ~20-22k tok/s dual.

NVFP4 (2.18 s micro-step, 30k dual) is the only measured configuration at the
target, but it diverges (NaN at step 2-3).

### BF16 round-2 follow-ups (all neutral or failing)

- ATEN GEMM autotune (`is_big_gpu` forced true, backends="ATEN"): 77 vs 78
  TFLOPS at 32768x1536x6144, 81 vs 81 at 8192^3. cuBLASLt's default heuristic
  is already the best available.
- CUTLASS autotune with `_normalize_cuda_arch` remapped to 100: every candidate
  fails (`Got cutlass error: Error Internal`, `min() iterable argument is empty`).
  No CUTLASS GEMM config exists for sm_121 in this torch build.
- `torch._inductor.config.epilogue_fusion = False`: 3.144 s vs 3.141 s. The
  sm80 CUTLASS relu-epilogue GEMMs (0.80 s, ~55 TFLOPS) are not replaced by
  cuBLASLt.
- `torch.mm` itself peaks at 81-83 TFLOPS, so the 1.89 s GEMM share is within
  8% of the hardware's demonstrated BF16 throughput.

Conclusion: the BF16 ceiling on this stack is ~20-21k tok/s dual-node. Reaching
30k would need a BF16 GEMM at >120 TFLOPS, which no available kernel provides
on sm_121, plus non-GEMM work at ~1/4 of today's cost.
