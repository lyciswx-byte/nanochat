// GB10 / SM121a: exact global amax followed by torchao-compatible 16x16
// block-scale NVFP4, row and column orientation, in ONE cooperative launch.
//
// One input load for amax, one for quantization: a mathematically necessary
// reread (or equally expensive staging) with tensor-wide exact scaling.
// Cooperative grid.sync avoids unsafe hand-rolled global spin barriers.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <float.h>
#include <climits>
#include <vector>
namespace cg = cooperative_groups;

constexpr int TM = 128, TN = 64, THREADS = 256;

// Do not silently mask NaNs: the base d24 NVFP4 run diverges after a
// few steps, and the amax output must reflect that to make it diagnosable.
__device__ __forceinline__ float max_nan(float a, float b) {
    return (isnan(a) || isnan(b)) ? nanf("") : fmaxf(a, b);
}


__device__ __forceinline__ float block_max(float v, float *shared) {
    for (int d = 16; d; d >>= 1)
        v = max_nan(v, __shfl_down_sync(0xffffffff, v, d));
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = v;
    __syncthreads();
    float w = threadIdx.x < (THREADS / 32) ? shared[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int d = 16; d; d >>= 1)
            w = max_nan(w, __shfl_down_sync(0xffffffff, w, d));
    }
    if (threadIdx.x == 0) shared[0] = w;
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_sum(float v, float *shared) {
    for (int d = 16; d; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = v;
    __syncthreads();
    float w = threadIdx.x < (THREADS / 32) ? shared[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int d = 16; d; d >>= 1)
            w += __shfl_down_sync(0xffffffff, w, d);
    }
    if (threadIdx.x == 0) shared[0] = w;
    __syncthreads();
    return shared[0];
}

// Round-to-nearest-even packed E2M1. No byte-expanded mxf8f6f4 detour.
__device__ __forceinline__ uint8_t fp4_pair(float low, float high) {
    unsigned int v;
    asm volatile(
        "{ .reg .b8 p;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 p, %1, %2;\n"
        "  cvt.u32.u8 %0, p;\n"
        "}\n"
        : "=r"(v) : "f"(high), "f"(low));
    return static_cast<uint8_t>(v);
}

__device__ __forceinline__ float fp8_decode(uint8_t b) {
    int e = (b >> 3) & 15;
    int m = b & 7;
    return e ? ldexpf(1.0f + m * (1.0f / 8.0f), e - 7)
             : ldexpf(m * (1.0f / 8.0f), -6);
}

__device__ __forceinline__ int sf_index(int outer, int inner, int inner_size) {
    // torchao _swizzle_scales reshape [OUTER/128,4,32,INNER/64,4]
    // then permute [0,3,2,1,4].
    return (((outer / 128) * (inner_size / 64) + inner / 64) * 32
            + (outer % 32)) * 16
            + (outer % 128) / 32 * 4 + (inner % 64) / 16;
}

template<int MODE>
__global__ void __launch_bounds__(THREADS)
quant_coop(const __nv_bfloat16 *__restrict__ x, int M, int N,
           uint8_t *__restrict__ row, uint8_t *__restrict__ row_sf,
           uint8_t *__restrict__ col, uint8_t *__restrict__ col_sf,
           float *__restrict__ output_amax, float *__restrict__ partial,
           float *__restrict__ row_rstd, float eps,
           __nv_bfloat16 *__restrict__ transformed_out) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __nv_bfloat16 tile[TM * TN];  // 16 KiB, <99 KiB GB10 limit.
    __shared__ float reduce_s[THREADS / 32];
    __shared__ float encode_s;
    __shared__ uint8_t scale_s;
    int tid = threadIdx.x;
    int64_t numel = int64_t(M) * N;

    float local = 0.0f;
    if constexpr (MODE == 1) {  // RMSNorm (no BF16 intermediate)
        for (int row_id = blockIdx.x; row_id < M; row_id += gridDim.x) {
            float sum_sq = 0.0f, max_abs = 0.0f;
            for (int j = tid; j < N; j += THREADS) {
                float v = __bfloat162float(x[int64_t(row_id) * N + j]);
                sum_sq += v * v;
                max_abs = max_nan(max_abs, fabsf(v));
            }
            float total = block_sum(sum_sq, reduce_s);
            float amax_row = block_max(max_abs, reduce_s);
            float rs = rsqrtf(total / N + eps);
            if (tid == 0) row_rstd[row_id] = rs;
            // BF16 rounding is monotone, so round(max*rs) == max(round(x*rs)).
            float y = fabsf(__bfloat162float(__float2bfloat16_rn(amax_row * rs)));
            local = max_nan(local, y);
        }
    } else {
        for (int64_t i = int64_t(blockIdx.x) * THREADS + tid; i < numel;
             i += int64_t(gridDim.x) * THREADS) {
            float v = __bfloat162float(x[i]);
            if constexpr (MODE == 2) {
                float relu = fmaxf(v, 0.0f);
                v = __bfloat162float(__float2bfloat16_rn(relu * relu));
            }
            local = max_nan(local, fabsf(v));
        }
    }
    float block_amax = block_max(local, reduce_s);
    if (tid == 0) partial[blockIdx.x] = block_amax;
    grid.sync();

    if (blockIdx.x == 0) {
        float m = 0.0f;
        for (int j = tid; j < gridDim.x; j += THREADS)
            m = max_nan(m, partial[j]);
        m = block_max(m, reduce_s);
        if (tid == 0) *output_amax = m;
    }
    grid.sync();

    float amax = *output_amax;
    float safe_amax = amax == 0.0f ? 1.0f : amax;
    float candidate = fminf(__fdiv_rn(2688.0f, safe_amax), FLT_MAX);
    if (candidate == 0.0f) candidate = 1.0f;
    float global_encode = amax == 0.0f ? 1.0f : candidate;
    float global_decode = __fdiv_rn(1.0f, global_encode);
    // The reference first scales BF16 block values with the tensor-wide
    // encode factor, then divides their maximum by FP4_MAX (6). Combining
    // these factors changes FP32 rounding at E4M3 halfway boundaries.

    int tiles_n = N / TN;
    int ntiles = (M / TM) * tiles_n;
    for (int tile_id = blockIdx.x; tile_id < ntiles; tile_id += gridDim.x) {
        int rbase = (tile_id / tiles_n) * TM;
        int cbase = (tile_id % tiles_n) * TN;
        // Every input value is loaded once during this second phase.
        for (int k = tid; k < TM * TN; k += THREADS) {
            int r = k / TN, c = k % TN;
            int64_t linear = int64_t(rbase + r) * N + cbase + c;
            __nv_bfloat16 y = x[linear];
            if constexpr (MODE == 1)
                y = __float2bfloat16_rn(__bfloat162float(y) * row_rstd[rbase + r]);
            if constexpr (MODE == 2) {
                float v = fmaxf(__bfloat162float(y), 0.0f);
                y = __float2bfloat16_rn(v * v);
            }
            tile[k] = y;
            if (transformed_out) transformed_out[linear] = y;
        }
        __syncthreads();

        // One warp owns one 16x16 group. Each warp reduces 256 BF16 values,
        // then emits both packed orientations and both swizzled scale atoms.
        // 8 warps x 4 rounds = all 32 groups, no inter-warp barriers.
        int lane = tid & 31;
        int warp = tid >> 5;
        for (int pass = 0; pass < 4; ++pass) {
            int group = pass * 8 + warp;
            int r0 = (group / (TN / 16)) * 16;
            int c0 = (group % (TN / 16)) * 16;
            float vmax = 0.0f;
            for (int k = lane; k < 256; k += 32) {
                float v = __bfloat162float(
                    tile[(r0 + k / 16) * TN + c0 + (k % 16)]);
                vmax = max_nan(vmax, fabsf(v));
            }
            for (int d = 16; d; d >>= 1)
                vmax = max_nan(vmax, __shfl_down_sync(0xffffffff, vmax, d));
            vmax = __shfl_sync(0xffffffff, vmax, 0);

            unsigned int sf = 0;
            float enc = 0.0f;
            if (lane == 0) {
                if (vmax == 0.0f) {
                    // torchao writes the E4M3 +0 byte, NOT the smallest
                    // normal/subnormal block scale, for an all-zero 16x16.
                    // Explicitly avoid 1 / (0 * global_decode) as well.
                    sf = 0;
                    enc = 0.0f;
                } else {
                    float scaled_max = vmax * global_encode;
                    float pvscale = fminf(__fdiv_rn(scaled_max, 6.0f), 448.0f);
                    sf = __nv_cvt_float_to_fp8(pvscale, __NV_SATFINITE, __NV_E4M3);
                    float denom = fp8_decode(uint8_t(sf)) * global_decode;
                    enc = fminf(__fdiv_rn(1.0f, denom), FLT_MAX);
                }
            }
            sf = __shfl_sync(0xffffffff, sf, 0);
            enc = __shfl_sync(0xffffffff, enc, 0);

            if (lane < 16) {
                int rr = rbase + r0 + lane;
                int cc = cbase + c0;
                row_sf[sf_index(rr, cc, N)] = uint8_t(sf);
                int tr = cbase + c0 + lane;
                int tc = rbase + r0;
                col_sf[sf_index(tr, tc, M)] = uint8_t(sf);
            }
            for (int k = lane; k < 128; k += 32) {
                int rr = r0 + k / 8;
                int cc = c0 + 2 * (k % 8);
                float x0 = __bfloat162float(tile[rr * TN + cc]) * enc;
                float x1 = __bfloat162float(tile[rr * TN + cc + 1]) * enc;
                row[int64_t(rbase + rr) * (N / 2) + (cbase + cc) / 2] =
                    fp4_pair(fminf(fmaxf(x0, -6.0f), 6.0f),
                             fminf(fmaxf(x1, -6.0f), 6.0f));

                int tr = c0 + k / 8;
                int tc = r0 + 2 * (k % 8);
                x0 = __bfloat162float(tile[tc * TN + tr]) * enc;
                x1 = __bfloat162float(tile[(tc + 1) * TN + tr]) * enc;
                col[int64_t(cbase + tr) * (M / 2) + (rbase + tc) / 2] =
                    fp4_pair(fminf(fmaxf(x0, -6.0f), 6.0f),
                             fminf(fmaxf(x1, -6.0f), 6.0f));
            }
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> fused_nvfp4_impl(
    torch::Tensor x, int mode, float eps, bool emit_bf16) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16,
                "fused_nvfp4 needs CUDA BF16");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "needs contiguous 2D input");
    int64_t M = x.size(0), N = x.size(1);
    TORCH_CHECK(M % 128 == 0 && N % 128 == 0, "dimensions must be multiples of 128");
    TORCH_CHECK(M <= INT_MAX && N <= INT_MAX, "dimensions exceed 32-bit tile indices");
    TORCH_CHECK(mode >= 0 && mode <= 2, "unrecognized transform mode");
    int device = x.get_device();
    cudaDeviceProp props;
    TORCH_CHECK(cudaGetDeviceProperties(&props, device) == cudaSuccess, "device query failed");
    TORCH_CHECK(props.major == 12 && props.minor == 1, "requires GB10 sm_121a");
    int cooperative = 0;
    TORCH_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device) == cudaSuccess
                && cooperative, "device does not support cooperative grid launch");
    const void *kernel = mode == 1 ? reinterpret_cast<const void *>(quant_coop<1>)
                        : mode == 2 ? reinterpret_cast<const void *>(quant_coop<2>)
                                    : reinterpret_cast<const void *>(quant_coop<0>);
    int blocks_per_sm = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, kernel, THREADS, 0) == cudaSuccess
                && blocks_per_sm > 0, "cooperative occupancy query failed");
    // Max 1 CTA/SM, guaranteed resident at grid.sync.
    int nblocks = props.multiProcessorCount;
    auto bytes = x.options().dtype(torch::kUInt8);
    auto row = torch::empty({M, N / 2}, bytes);
    auto row_sf = torch::empty({M / 128, N / 64, 32, 16},
                               x.options().dtype(at::kFloat8_e4m3fn));
    auto col = torch::empty({N, M / 2}, bytes);
    auto col_sf = torch::empty({N / 128, M / 64, 32, 16},
                               x.options().dtype(at::kFloat8_e4m3fn));
    auto amax = torch::empty({}, x.options().dtype(torch::kFloat32));
    auto scratch = torch::empty({nblocks}, x.options().dtype(torch::kFloat32));
    auto rstd = torch::empty({mode == 1 ? M : 0}, x.options().dtype(torch::kFloat32));
    auto bf16 = torch::empty(emit_bf16 ? std::vector<int64_t>{M, N}
                                      : std::vector<int64_t>{0}, x.options());
    auto *xp = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
    auto *rp = row.data_ptr<uint8_t>(), *cp = col.data_ptr<uint8_t>();
    auto *rsp = reinterpret_cast<uint8_t *>(row_sf.data_ptr());
    auto *csp = reinterpret_cast<uint8_t *>(col_sf.data_ptr());
    auto *amp = amax.data_ptr<float>(), *pp = scratch.data_ptr<float>();
    auto *rstdp = mode == 1 ? rstd.data_ptr<float>() : (float *)nullptr;
    auto *outp = emit_bf16 ? reinterpret_cast<__nv_bfloat16 *>(bf16.data_ptr())
                            : (__nv_bfloat16 *)nullptr;
    int mi = int(M), ni = int(N);
    void *args[] = {&xp, &mi, &ni, &rp, &rsp, &cp, &csp, &amp, &pp,
                    &rstdp, &eps, &outp};
    auto stream = at::cuda::getCurrentCUDAStream(device).stream();
    auto err = cudaLaunchCooperativeKernel(const_cast<void *>(kernel),
                                           dim3(nblocks), dim3(THREADS),
                                           args, 0, stream);
    TORCH_CHECK(err == cudaSuccess, "cooperative NVFP4 launch failed: ",
                cudaGetErrorString(err));
    return {row, row_sf, col, col_sf, amax, bf16};
}

std::vector<torch::Tensor> fused_nvfp4_quant_cuda(torch::Tensor x) {
    auto result = fused_nvfp4_impl(x, 0, 0.0f, false);
    result.pop_back();
    return result;
}

std::vector<torch::Tensor> fused_nvfp4_rmsnorm_cuda(
    torch::Tensor x, double eps, bool return_bf16) {
    return fused_nvfp4_impl(x, 1, float(eps), return_bf16);
}

std::vector<torch::Tensor> fused_nvfp4_relu2_cuda(
    torch::Tensor x, bool return_bf16) {
    return fused_nvfp4_impl(x, 2, 0.0f, return_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &fused_nvfp4_quant_cuda, "GB10 cooperative plain NVFP4");
    m.def("rmsnorm_quant", &fused_nvfp4_rmsnorm_cuda, "GB10 RMSNorm->NVFP4");
    m.def("relu2_quant", &fused_nvfp4_relu2_cuda, "GB10 ReLU2->NVFP4");
}
