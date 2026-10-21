// Fused BF16 GEMM with a ReLU^2 epilogue for sm_121 (via CUTLASS 2.x Sm80
// tensorop kernels, which run on sm_121). C = relu(A @ B^T)^2.
//
// A: (M, K) row-major BF16, B: (N, K) row-major BF16, C: (M, N) row-major BF16.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

namespace {

template <typename T>
struct ReluSquare {
    static const bool kIsHeavy = false;
    CUTLASS_HOST_DEVICE
    T operator()(T v) const { return v > T(0) ? v * v : T(0); }
};

template <typename Element, int N>
struct ReluSquare<cutlass::Array<Element, N>> {
    static const bool kIsHeavy = false;
    CUTLASS_HOST_DEVICE
    cutlass::Array<Element, N> operator()(cutlass::Array<Element, N> v) const {
        for (int i = 0; i < N; ++i) {
            v[i] = v[i] > Element(0) ? v[i] * v[i] : Element(0);
        }
        return v;
    }
};

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAcc = float;

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGeneric<
    ReluSquare, ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc, ElementAcc>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, cutlass::layout::RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

}  // namespace

torch::Tensor relu2_gemm(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "BF16 tensors required");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "2D tensors required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous required");
    const int64_t M = a.size(0), K = a.size(1), N = b.size(0);
    TORCH_CHECK(b.size(1) == K, "shape mismatch");
    auto c = torch::empty({M, N}, a.options());

    Gemm gemm;
    Gemm::Arguments args(
        cutlass::gemm::GemmCoord(int(M), int(N), int(K)),
        {reinterpret_cast<ElementA const *>(a.data_ptr()), int(K)},
        {reinterpret_cast<ElementB const *>(b.data_ptr()), int(K)},
        {reinterpret_cast<ElementC *>(c.data_ptr()), int(N)},
        {reinterpret_cast<ElementC *>(c.data_ptr()), int(N)},
        {1.0f, 0.0f});
    auto status = gemm.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "relu2 gemm cannot implement: ", int(status));
    status = gemm.initialize(args, nullptr, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "relu2 gemm init failed: ", int(status));
    status = gemm();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "relu2 gemm run failed: ", int(status));
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu2_gemm", &relu2_gemm, "BF16 GEMM with ReLU^2 epilogue");
}
