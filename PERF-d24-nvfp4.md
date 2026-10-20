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
