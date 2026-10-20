// BF16 row-major GEMM through cublasGemmEx (legacy cuBLAS path).
//
// PyTorch's matmul reaches ~77-79 TFLOPS at the d24 shapes on GB10 while a
// direct cublasGemmEx call reaches ~87-89 TFLOPS for the same operands.
// Exposed shapes:
//   linear(x[M,K], w[N,K]) -> y[M,N] = x @ w^T          (NT)
//   dgrad(dy[M,N], w[N,K]) -> dx[M,K] = dy @ w          (NN)
//   wgrad(dy[M,N], x[M,K]) -> dw[N,K] = dy^T @ x
// All operands are contiguous BF16, accumulation in FP32.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

namespace {

cublasHandle_t handle_for(int device) {
    static cublasHandle_t handles[16] = {nullptr};
    if (handles[device] == nullptr) {
        TORCH_CHECK(cublasCreate(&handles[device]) == CUBLAS_STATUS_SUCCESS,
                    "cublasCreate failed");
    }
    return handles[device];
}

inline void set_stream(cublasHandle_t h) {
    cublasSetStream(h, at::cuda::getCurrentCUDAStream());
}

inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                           int m, int n, int k,
                           const void *a, int lda, const void *b, int ldb,
                           void *c, int ldc) {
    float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(h, ta, tb, m, n, k, &alpha,
                        a, CUDA_R_16BF, lda,
                        b, CUDA_R_16BF, ldb,
                        &beta, c, CUDA_R_16BF, ldc,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

}  // namespace

torch::Tensor linear_bf16(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
                "BF16 tensors required");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "2D tensors required");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "contiguous tensors required");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K, "shape mismatch");
    auto y = torch::empty({M, N}, x.options());
    auto h = handle_for(x.get_device());
    set_stream(h);
    // row-major C[M,N] = A[M,K] @ B[N,K]^T  ==  col-major C^T[N,M] = B @ A^T
    auto st = gemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K,
                   w.data_ptr(), (int)K, x.data_ptr(), (int)K,
                   y.data_ptr(), (int)N);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublas linear failed: ", (int)st);
    return y;
}

torch::Tensor dgrad_bf16(torch::Tensor dy, torch::Tensor w) {
    const int64_t M = dy.size(0), N = dy.size(1), K = w.size(1);
    TORCH_CHECK(w.size(0) == N, "shape mismatch");
    auto dx = torch::empty({M, K}, dy.options());
    auto h = handle_for(dy.get_device());
    set_stream(h);
    // row-major C[M,K] = dy[M,N] @ w[N,K]  ==  col-major C^T[K,M] = w^T @ dy^T
    auto st = gemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)K, (int)M, (int)N,
                   w.data_ptr(), (int)K, dy.data_ptr(), (int)N,
                   dx.data_ptr(), (int)K);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublas dgrad failed: ", (int)st);
    return dx;
}

torch::Tensor wgrad_bf16(torch::Tensor dy, torch::Tensor x) {
    const int64_t M = x.size(0), K = x.size(1), N = dy.size(1);
    TORCH_CHECK(dy.size(0) == M, "shape mismatch");
    auto dw = torch::empty({N, K}, x.options());
    auto h = handle_for(x.get_device());
    set_stream(h);
    // row-major C[N,K] = dy^T[N,M] @ x[M,K]  ==  col-major C^T[K,N] = x^T @ dy
    auto st = gemm(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)K, (int)N, (int)M,
                   x.data_ptr(), (int)K, dy.data_ptr(), (int)N,
                   dw.data_ptr(), (int)K);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublas wgrad failed: ", (int)st);
    return dw;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &linear_bf16, "BF16 linear (x @ w^T)");
    m.def("dgrad", &dgrad_bf16, "BF16 dgrad (dy @ w)");
    m.def("wgrad", &wgrad_bf16, "BF16 wgrad (dy^T @ x)");
}
