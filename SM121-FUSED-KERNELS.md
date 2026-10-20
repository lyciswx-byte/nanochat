# SM121a fused NVFP4 kernels — experimental

Base: dgx-spark at 57d056b4f056b24c3ef33ade40b6430286b087e7.
Target: GB10, CUDA 13.0, PyTorch 2.14.0a0, Triton 3.8.0, torchao 0.19.0.

## Architecture

The CUDA 13 quantization extension targets **sm_121a**, uses 256 threads (8 warps) per cooperative CTA, a 128 x 64 BF16 input tile, and **16 KiB of shared memory**. It contains no tcgen05, TMEM or 2-SM instructions.

Phase 1 reads input and computes partial maxima, synchronizes the cooperative grid, and computes the device-wide global amax. Phase 2 rereads the input, with one warp per 16x16 quantization block. Each warp emits round-to-nearest-even E2M1 packed codes (2 values/byte) and per-block E4M3 scales in torchao's exact 32x16 swizzled layout, for both row and column orientations. The GEMMs remain the existing torchao native packed FP4 scaled_mm path using warp-level m16n8k64 mxf4nvf4 MMA.

**One launch does not mean one input read.** Exact global scaling depends on an amax computed over the entire tensor. The implementation therefore performs two input reads and one cooperative launch. A literal one-read exact scheme would need to retain or stage the entire input pending global reduction. There is no magical sublinear-storage solution to this dependency.

For RMSNorm and ReLU-squared transformations, combining them with quantization removes additional BF16 intermediate traffic. Forward fused RMSNorm first determines each row's reciprocal RMS in parallel while computing the global max of BF16-rounded normalized values; the second phase quantizes those values without writing BF16 unless explicitly requested. The MLP pre-fc1 fused norm path has its own RMSNorm autograd backward.

## Opt-in flags (default off)

- NANOCHAT_NVFP4_FUSED_QUANT=1 — plain activation amax and quantization fused for all three torchao GEMMs, including backward.
- NANOCHAT_NVFP4_FUSED_RMSNORM=1 — fuse the **24 MLP pre-fc1 RMSNorms** into packed operands and the fc1 GEMM. The attention-side 24 norms still have other consumers, and remain unchanged.
- NANOCHAT_NVFP4_FUSED_MLP=1 — quantize ReLU^2 directly into fc2 operands, eliminating the **additional** BF16 squared-activation tensor. The fc1 GEMM still materializes its BF16 output: a CUTLASS-integrated GEMM epilogue is not implemented.
- NANOCHAT_FUSED_VERBOSE=1 — verbose NVCC JIT build messages.

All three flags require plain quantization (use_rht=False). Attempting fused kernels with RHT enabled explicitly errors. The fast fused-RHT transform has not been implemented. No persistent parameters or checkpoint keys are changed. Opaque custom operators allocate fresh outputs and register matching FakeTensor shapes. The extension JIT compilation and warm kernel launch occur before graph capture.

## Reproducible bring-up (NGC 26.09, GB10)

    python -m pytest -q tests/test_fused_nvfp4.py
    python bench_kernels.py --check-only
    python bench_kernels.py --warmup 3 --iters 10
    NANOCHAT_NVFP4_FUSED_QUANT=1 python bench_kernels.py --autograd-smoke --skip-large

Bench dimensions are M=32768, (K,N)=(1536,4608), (1536,6144), (6144,1536), (32768,1536). Both (M,K) activations and (M,N) gradients are measured, plus E4M3 scale/code and dequantization error versus torchao. Event timing reports milliseconds, estimated minimum memory traffic, effective GB/s and speedups.

Quant-only minimum traffic: two BF16 reads, two 4-bit packed output orientations, and two sets of 1-byte-per-16-element scale arrays, or 5.125 bytes per logical input element. Fused norm/activation skips at least the BF16 transformed tensor's 2-byte write and subsequent read. Exact 60k tokens/s requires end-to-end profiling, not just operator timings.

## Unverified on-device requirements

1. Confirm CUDA 13 extension builds with native sm_121a, cooperative launch is supported, and CUDA Graph capture/replay succeeds on GB10.
2. Check packed nibble order, per-16x16 scale rounding, transposed codes, torchao swizzles and global tensor scale on multiple shapes, zeros and saturation cases.
3. Validate forward and backward equivalence under torch.compile, weight cache ownership and CUDA Graph replay.
4. Test stability over more than four optimizer steps on both GB10s. The original base commit records NaNs in dual-node NVFP4 **with RHT**, too.
5. Profile 200 GB/s sustainable bandwidth, kernel ms and 2-node tok/s against the original torchao path.

**Status:** implementation written through GitHub connector; not built, run, profiled, or numerically validated on physical GPUs. No performance result is claimed.
