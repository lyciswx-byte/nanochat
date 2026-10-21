# Codex task: optimize the d24 BF16 training step on 2x DGX Spark (GB10)

You are a senior GPU performance engineer working **on this machine** (you have a
shell). The repo is `/media/lycis/data/nanochat`. The GPU work runs inside the
Docker container `nanochat-train` (NGC PyTorch 26.09). Only one GPU exists, so
never run two GPU jobs at once.

Work on a new branch `codex/bf16-opt-<yyyymmdd-hhmm>` from the current tip of
`bf16-perf` (verify with `git ls-remote origin refs/heads/bf16-perf`; it already
contains this file, `bench_parts.py`, `prof_bf16.py` and `run_train_bf16.sh`).
Never force-push, never push to `bf16-perf` or `dgx-spark` directly. Commit
small, one change per commit, with the measured numbers in the message.

## Mission

Raise the two-node d24 BF16 training throughput from the measured **22.6k tok/s**
toward **24-25k tok/s** without changing model math. The hard target of 30k has
been shown to need ~100 TFLOPS sustained GEMM plus a ~2x cut in non-GEMM time and
is out of scope for this task; do not chase it by changing precision (FP8/FP4)
or the model.

## Hardware / stack (measured facts)

- 2x NVIDIA GB10, sm_121a, 48 SMs, ~2.4 GHz, 24 MiB L2, 121 GB unified memory.
- Memory: 236 GB/s read, 216 GB/s copy; 200 Gbps RoCE between nodes.
- torch 2.14.0a0 (NGC 26.09), CUDA 13.4, Triton 3.8.0, torchao 0.19.0,
  flash-attn 2.8.3, Python 3.12. CUTLASS 4 headers at
  `/opt/pytorch/pytorch/third_party/cutlass/include` (2.x API available).
- GB10 has **no tcgen05/TMEM/2-SM MMA**; only warp-level `mma.sync`, 99 KiB smem.

## Current best configuration (validated, reproducible)

```bash
docker exec nanochat-train bash -lc 'cd /workspace/nanochat && \
  NANOCHAT_CUBLAS_LINEAR=1 NANOCHAT_MASTER_BF16=1 \
  python3 -m scripts.base_train --depth=24 --device-batch-size=16 \
  --window-pattern=SSSL --run=dummy --save-every=1000 --eval-every=-1 \
  --core-metric-every=-1 --sample-every=-1 --num-iterations=4 \
  --model-tag=bf16opt'
```

Measured: **2.775-2.809 s per GPU micro-step** (run-to-run spread ~35 ms),
**11.3k tok/s/GPU -> 22.6k dual**, peak allocation 53.0 GiB, four-step loss
10.397344 / 10.371974 / 10.322719 / 10.252465 (bit-identical to the plain BF16
baseline).

What `NANOCHAT_CUBLAS_LINEAR=1` does: `Linear.forward` sends every Linear except
the MLP up-projection through an opaque `cublasGemmEx` op
(`nanochat/kernels/cublas_linear.py` + `csrc/cublas_linear.cu`, fwd NT, dgrad,
wgrad, bit-exact vs `F.linear`). The MLP `c_fc` keeps `F.linear`
(`self.c_fc._keep_fused = True`) because Inductor fuses its ReLU into the GEMM
epilogue.

Micro-step profile at this config: total 2.81 s; **GEMM 1.94 s** and non-GEMM
0.87 s. Model matmul volume is **165.7 TFLOP** per micro-step (fwd 55.2 + bwd
110.5), i.e. **85.2 TFLOPS aggregate**. Non-GEMM: attention 0.31 s (FA2),
ReLU^2 0.24 s, RMSNorm ~0.40 s, embeddings/CE/add/misc ~0.30 s.

Operator-level GEMM ceiling measured with a `cublasGemmAlgo_t` 0..23 sweep
(30 runs): c_fc 94.7, dgrad 92.1, mlp c_proj 91.9, lm_head 93.3, wgrad 89.9,
qkv 85.8 TFLOPS. So **92+ is reachable per operator**; the model sustains 85.2.

## What has already been tried (do NOT repeat without a new mechanism)

| Experiment | Result |
|---|---|
| All Linear through cuBLAS (no `_keep_fused`) | 3.141 s, neutral: loses the relu epilogue |
| Per-shape `cublasGemmAlgo_t` table in-model | 2.805 s, neutral |
| cuBLASLt `CUBLASLT_EPILOGUE_RELU` for MLP | 2.805 s, neutral |
| Triton 4-row RMSNorm backward (opaque) | 2.863 s (+88 ms), NO-GO |
| Chunked CUTLASS relu^2 GEMM integrated | 3.036 s (+231 ms), NO-GO (16 launches + `torch.cat` copies + lost fusion) |
| FA2 64x64 tile for full layers | not testable without rebuilding flash-attn; cuDNN SDPA fwd 2.56 ms vs FA2 2.53 ms |
| `mode=reduce-overhead` / `max-autotune` | 3.148 / 3.210 s; autotune disabled below 68 SMs |
| Forced CUTLASS autotune | every config errors on sm_121 |
| Fused QKV, epilogue_fusion=False, batch 24 | neutral / +4% at 72 GiB |

## Remaining hypotheses (ranked; each needs a paired A/B)

1. **Fused relu^2 GEMM with out+offset writes.** The experimental kernel
   `nanochat/kernels/csrc/relu2_gemm.cu` is correct (5e-4) and reaches 86 TFLOPS
   when M is chunked to 2048 rows, but integrating it lost 231 ms because the
   output is `torch.cat`-ed from 16 parts. Add an `out`/row-offset argument so
   chunks write directly into one buffer, fuse the square's backward
   (`g = 2*sqrt(y2)*dout`) and re-measure. GO if the step improves by >=40 ms.
2. **Fix the L2-cliff without chunking.** One call at M=32768 collapses to
   21 TFLOPS because identity threadblock order streams A once per N tile.
   `GemmHorizontalThreadblockSwizzle` fails to compile with this CUTLASS 4
   (kernel wants `get_tiled_shape()`). Options: implement a compatible swizzle,
   use a CUTLASS 3.x collective with an sm120 schedule, or a persistent kernel.
   GO if one-shot >=85 TFLOPS.
3. **Non-GEMM fusions that preserve Inductor.** The RMSNorm attempt failed only
   because the op was opaque. Look for fusions that stay visible to Inductor
   (e.g. `triton.coordinate_descent_tuning`, `multi_kernel`, custom lowerings,
   or a fused residual+norm that Inductor can still merge downstream). GO if the
   RMSNorm group drops by >=40 ms with unchanged loss.
4. **Attention.** FA2 uses 128x64 tiles; a 64x64 causal tile may help the 6
   full-window layers, but needs a flash-attn source rebuild. GO if >=40 ms.
5. **Distributed overhead.** Ideal dual-node from 2.775 s would be 23.6k; we see
   22.6k, i.e. ~0.1 s of effective overhead. Try DDP `static_graph`,
   `gradient_as_bucket_view`, comm/compute overlap, or bucketed all-reduce.
   GO if >=50 ms.

## Hard constraints and traps (learned the hard way)

- **Never return a persistent cached tensor from an opaque custom op.** Inductor
  owns custom-op outputs and donates/frees their storage. That bug silently
  overwrote packed weights: a compiled forward repeated with unchanged weights
  jumped from loss 10.3976 to 24.81 and real training produced NaN. Return
  freshly allocated tensors (clones are acceptable).
- **Never rely on a non-persistent buffer as a shadow of a parameter.** Inductor
  bakes it into the graph; copying new master values after `optimizer.step()`
  never reaches the forward, and training diverges to NaN.
- Opaque operator boundaries break Inductor's residual/norm/relu epilogue
  fusion. Every new op must be A/B tested end to end, not just in isolation.
- Preserve the four-step loss trajectory (tolerance 1e-3) and the 110 GB host
  memory limit. `torch.cuda.max_memory_allocated()` alone is not host RSS.
- No new persistent parameters; checkpoints must stay compatible.
- GPU jobs must run alone. Kill leftovers with
  `docker exec nanochat-train bash -lc 'pkill -f scripts.base_train'` and check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before timing.

## Measurement protocol

- Baseline and candidate must be measured in the same session with the same
  `bench_parts.py` (`PREC=bf16 MASTER=bf16 BATCH=16 DEPTH=24 ITERS=5`) and with
  the full trainer for the final number. Report median of 5 iterations, and
  repeat the baseline at least twice because the run-to-run spread is ~35 ms.
- Correctness: compare 4-step loss against the baseline above; for new GEMM
  operators also check max abs error versus `F.linear` (expect <=1e-3 relative
  on the pre-activation, or bit-exact where applicable).
- Report peak allocation and the host `free -g` peak.

## Deliverables

1. Branch with one commit per accepted change, each message containing the
   before/after micro-step and the GO/NO-GO decision.
2. A short `BENCH-<name>.md` with the paired measurements and the exact command.
3. A final summary: branch, SHAs, what was kept, what was rejected and why.

Start by reproducing the current best (2.775-2.809 s) and the profile above, then
pick hypothesis 1 or 2 and run one tightly scoped experiment. Do not batch
several unproven changes into one measurement.
