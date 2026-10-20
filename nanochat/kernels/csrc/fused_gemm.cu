// sm_121a warp-level NVFP4 GEMM. See lyciswx-byte/nvfp4bench/src/gemm_warp_mxf4.cu
// and NVIDIA/cutlass/include/cute/{arch/mma_sm120.hpp,atom/mma_traits_sm120.hpp}.
// CUDA 13.0 required. No tcgen05, TMEM, multicast, or multi-SM MMA.
// Canonical packed layout: [rows, K/2], low nibble = even K.
// Canonical SF layout: [rows, K/16] uint8 E4M3FN, unswizzled.
// Every output is freshly allocated; no pointer- or weight-cache ownership escapes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <tuple>
#include <cmath>

namespace {
constexpr int BM=64, BN=64, BK=64;
constexpr int WARPS_M=4, WARPS_N=2, THREADS=256;
constexpr float DENOM=2688.0f;  // E2M1 max * E4M3FN max = 6 * 448
using bf16 = __nv_bfloat16;

__device__ __forceinline__ float readbf(const bf16* ptr) {
  return __bfloat162float(*ptr);
}
__device__ __forceinline__ uint8_t fp8code(float x) {
  return __nv_cvt_float_to_fp8(fminf(fmaxf(x, 0.f),448.f), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float fp8decode(uint8_t x) {
  __nv_fp8_e4m3 v;
  v.__x = x;
  return float(v);
}
__device__ __forceinline__ uint8_t e2m1code(float x) {
  const float mag[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  float ax=fminf(fabsf(x), 6.f);
  int best=0;
  float err=fabsf(ax-mag[0]);
#pragma unroll
  for(int c=1;c<8;++c) {
    float e=fabsf(ax-mag[c]);
    if(e < err || (e == err && (c&1)==0 && (best&1)!=0)) {
      best=c; err=e;
    }
  }
  return uint8_t(best | (signbit(x) ? 8 : 0));
}
__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1])<<8) |
         (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24);
}
__device__ __forceinline__ void mma_f4(float (&dst)[4], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2],
                                        uint32_t sfa, uint32_t sfb) {
  const float c0=dst[0],c1=dst[1],c2=dst[2],c3=dst[3];
  const uint16_t bid=0, tid=0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  // This is genuine per-16 E4M3 (NOT the benchmark's per-32 UE8M0 shortcut).
  // Scale layouts: A row=8*(lane&1)+(lane>>2), B row=lane>>2;
  // each 32-bit SF word contains four consecutive K/16 scales.
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col."
    "f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(dst[0]),"=f"(dst[1]),"=f"(dst[2]),"=f"(dst[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
      "r"(b[0]),"r"(b[1]),"f"(c0),"f"(c1),"f"(c2),"f"(c3),
      "r"(sfa),"h"(bid),"h"(tid),"r"(sfb),"h"(bid),"h"(tid));
#else
  // Host compilation has no device execution. The launch rejects other GPUs.
  (void)dst;(void)a;(void)b;(void)sfa;(void)sfb;
#endif
}

// Each CTA quantizes only its own BF16 input fragments, never publishing a
// packed A tensor to global memory. Global maxima are kernel *inputs*.
template<bool PACKED_B, bool OUT_FP4>
__global__ void gemm(const bf16* __restrict__ A,
                     const bf16* __restrict__ B,
                     const uint8_t* __restrict__ Bp,
                     const uint8_t* __restrict__ Bsf,
                     const float* __restrict__ amaxA,
                     const float* __restrict__ amaxB,
                     const float* __restrict__ Bglobal,
                     const float* __restrict__ outAmax,
                     bf16* __restrict__ Out,
                     uint8_t* __restrict__ Outp,
                     uint8_t* __restrict__ Outsf,
                     int M,int N,int K) {
  __shared__ __align__(16) uint8_t As[BM*BK/2];
  __shared__ __align__(16) uint8_t Bs[BN*BK/2];
  __shared__ __align__(16) uint8_t Sfa[BM*(BK/16)];
  __shared__ __align__(16) uint8_t Sfb[BN*(BK/16)];
  // Used only by OUT_FP4; keeping it static simplifies CUDA graph capture.
  __shared__ float Ctile[BM*BN];

  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  const int wm=warp/WARPS_N, wn=warp%WARPS_N;
  const int g=lane>>2, t=lane&3;
  const int mb=blockIdx.y*BM, nb=blockIdx.x*BN;
  const float ga=(amaxA[0]>0 && isfinite(amaxA[0]))? amaxA[0]/DENOM:1.f;
  const float gb=PACKED_B ? Bglobal[0] :
      ((amaxB[0]>0 && isfinite(amaxB[0]))? amaxB[0]/DENOM:1.f);
  float acc[4][4];
#pragma unroll
  for(int i=0;i<4;++i) for(int j=0;j<4;++j) acc[i][j]=0.f;

  for(int kb=0;kb<K;kb+=BK) {
    // 256 independent 16-element blocks on A, then B. Each group is
    // owned by one thread, preventing duplicate / racing packed-byte stores.
    {
      const int row=tid>>2, q=tid&3;
      float x[16], maxabs=0.f;
#pragma unroll
      for(int i=0;i<16;++i) {
        x[i]=readbf(A+size_t(mb+row)*K+kb+q*16+i);
        maxabs=fmaxf(maxabs,fabsf(x[i]));
      }
      uint8_t s=fp8code(maxabs/(6.f*ga));
      if(maxabs>0.f && s==0) s=1;
      Sfa[row*4+q]=s;
      const float scale=fp8decode(s)*ga;
#pragma unroll
      for(int j=0;j<8;++j) {
        uint8_t lo=scale>0.f?e2m1code(x[2*j]/scale):0;
        uint8_t hi=scale>0.f?e2m1code(x[2*j+1]/scale):0;
        As[row*32+q*8+j]=uint8_t(lo|(hi<<4));
      }
    }
    {
      const int row=tid>>2, q=tid&3;
      if constexpr(PACKED_B) {
        Sfb[row*4+q]=Bsf[size_t(nb+row)*(K/16)+kb/16+q];
#pragma unroll
        for(int j=0;j<8;++j)
          Bs[row*32+q*8+j]=Bp[size_t(nb+row)*(K/2)+kb/2+q*8+j];
      } else {
        float x[16], maxabs=0.f;
#pragma unroll
        for(int i=0;i<16;++i) {
          x[i]=readbf(B+size_t(nb+row)*K+kb+q*16+i);
          maxabs=fmaxf(maxabs,fabsf(x[i]));
        }
        uint8_t s=fp8code(maxabs/(6.f*gb));
        if(maxabs>0.f && s==0) s=1;
        Sfb[row*4+q]=s;
        const float scale=fp8decode(s)*gb;
#pragma unroll
        for(int j=0;j<8;++j) {
          uint8_t lo=scale>0.f?e2m1code(x[2*j]/scale):0;
          uint8_t hi=scale>0.f?e2m1code(x[2*j+1]/scale):0;
          Bs[row*32+q*8+j]=uint8_t(lo|(hi<<4));
        }
      }
    }
    __syncthreads();

    const int r0=wm*16+2*g, r1=r0+1;
    const uint32_t af[4]={
      ld4(As+r0*32+t*8), ld4(As+r1*32+t*8),
      ld4(As+r0*32+t*8+4),ld4(As+r1*32+t*8+4)
    };
    const int sf_row=wm*16+8*(lane&1)+(lane>>2);
    const uint32_t sfA=ld4(Sfa+sf_row*4);
#pragma unroll
    for(int ni=0;ni<4;++ni) {
      const int col=wn*32+ni*8+g;
      const uint32_t bf[2]={
        ld4(Bs+col*32+t*8),ld4(Bs+col*32+t*8+4)
      };
      const uint32_t sfB=ld4(Sfb+col*4);
      mma_f4(acc[ni],af,bf,sfA,sfB);
    }
    __syncthreads();
  }
  const float global=ga*gb;
#pragma unroll
  for(int ni=0;ni<4;++ni) {
    const int r=wm*16+2*g,c=wn*32+ni*8+2*t;
    const float v0=acc[ni][0]*global,v1=acc[ni][1]*global;
    const float v2=acc[ni][2]*global,v3=acc[ni][3]*global;
    if constexpr(OUT_FP4) {
      Ctile[(r+0)*BN+c+0]=v0; Ctile[(r+0)*BN+c+1]=v1;
      Ctile[(r+1)*BN+c+0]=v2; Ctile[(r+1)*BN+c+1]=v3;
    } else {
      Out[size_t(mb+r)*N+nb+c]=__float2bfloat16_rn(v0);
      Out[size_t(mb+r)*N+nb+c+1]=__float2bfloat16_rn(v1);
      Out[size_t(mb+r+1)*N+nb+c]=__float2bfloat16_rn(v2);
      Out[size_t(mb+r+1)*N+nb+c+1]=__float2bfloat16_rn(v3);
    }
  }
  if constexpr(OUT_FP4) {
    __syncthreads();
    const int r=tid/4, q=tid%4;
    float mx=0.f;
#pragma unroll
    for(int i=0;i<16;++i) mx=fmaxf(mx,fabsf(Ctile[r*BN+q*16+i]));
    const float gout=(outAmax[0]>0.f && isfinite(outAmax[0]))?
          outAmax[0]/DENOM:1.f;
    uint8_t s=fp8code(mx/(6.f*gout));
    if(mx>0.f && s==0) s=1;
    Outsf[size_t(mb+r)*(N/16)+(nb/16)+q]=s;
    const float scale=fp8decode(s)*gout;
#pragma unroll
    for(int j=0;j<8;++j) {
      const uint8_t a=scale>0.f?e2m1code(Ctile[r*BN+q*16+2*j]/scale):0;
      const uint8_t b=scale>0.f?e2m1code(Ctile[r*BN+q*16+2*j+1]/scale):0;
      Outp[size_t(mb+r)*(N/2)+nb/2+q*8+j]=uint8_t(a|(b<<4));
    }
  }
}

void check_common(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& amax, bool packed) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && amax.is_cuda(), "CUDA inputs required");
  TORCH_CHECK(a.scalar_type()==at::kBFloat16 && a.dim()==2 && a.is_contiguous(), "A must be contiguous BF16 [M,K]");
  TORCH_CHECK(b.dim()==2 && b.is_contiguous(), "B must be contiguous [N,K] or [N,K/2]");
  TORCH_CHECK(a.device()==b.device() && a.device()==amax.device(),"device mismatch");
  TORCH_CHECK(amax.scalar_type()==at::kFloat && amax.numel()==1,"amax must be one FP32 element");
  TORCH_CHECK(a.size(0)>0 && a.size(1)>0 && a.size(0)%BM==0 && a.size(1)%BK==0,
              "M and K must be positive multiples of 64");
  TORCH_CHECK(b.size(0)>0 && b.size(0)%BN==0,"N must be a positive multiple of 64");
  TORCH_CHECK(b.size(1)==a.size(1)/(packed?2:1),"B K shape mismatch");
  if(packed) TORCH_CHECK(b.scalar_type()==at::kByte,"packed B must be uint8");
  else TORCH_CHECK(b.scalar_type()==at::kBFloat16,"B must be BF16");
  cudaDeviceProp props{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&props,a.get_device()));
  TORCH_CHECK(props.major==12 && props.minor==1,"sm_121 GPU required");
}
void check_scalar(const torch::Tensor& x,const torch::Tensor& a,const char* label) {
  TORCH_CHECK(x.is_cuda() && x.device()==a.device() && x.numel()==1 &&
              x.scalar_type()==at::kFloat && x.is_contiguous(),label,
              " must be one contiguous FP32 CUDA scalar");
}
void check_packed(const torch::Tensor& a,const torch::Tensor& bp,
                  const torch::Tensor& sf,const torch::Tensor& bglobal) {
  check_common(a,bp,torch::empty({1},a.options().dtype(at::kFloat)),true);
  TORCH_CHECK(sf.is_cuda() && sf.device()==a.device() && sf.scalar_type()==at::kByte &&
              sf.is_contiguous() && sf.dim()==2 && sf.size(0)==bp.size(0) &&
              sf.size(1)==a.size(1)/16, "B scales must be uint8 [N,K/16]");
  check_scalar(bglobal,a,"B global scale");
}
template<bool PACKED_B,bool OUT_FP4>
void launch(const torch::Tensor& a,const torch::Tensor& b,
            const torch::Tensor& sf,const torch::Tensor& amax,
            const torch::Tensor& bmax,const torch::Tensor& bglobal,
            const torch::Tensor& outmax,torch::Tensor& out,
            torch::Tensor& code,torch::Tensor& osf) {
  c10::cuda::CUDAGuard guard(a.device());
  dim3 grid(b.size(0)/BN,a.size(0)/BM);
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  gemm<PACKED_B,OUT_FP4><<<grid,THREADS,0,stream>>>(
     reinterpret_cast<const bf16*>(a.data_ptr<at::BFloat16>()),
     PACKED_B?nullptr:reinterpret_cast<const bf16*>(b.data_ptr<at::BFloat16>()),
     PACKED_B?b.data_ptr<uint8_t>():nullptr,
     PACKED_B?sf.data_ptr<uint8_t>():nullptr,
     amax.data_ptr<float>(),
     PACKED_B?nullptr:bmax.data_ptr<float>(),
     PACKED_B?bglobal.data_ptr<float>():nullptr,
     OUT_FP4?outmax.data_ptr<float>():nullptr,
     OUT_FP4?nullptr:reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
     OUT_FP4?code.data_ptr<uint8_t>():nullptr,
     OUT_FP4?osf.data_ptr<uint8_t>():nullptr,
     a.size(0),b.size(0),a.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor forward_bf16(torch::Tensor a,torch::Tensor b,
                           torch::Tensor amax,torch::Tensor bmax) {
  check_common(a,b,amax,false);check_scalar(bmax,a,"B amax");
  auto out=torch::empty({a.size(0),b.size(0)},a.options());
  torch::Tensor empty;
  launch<false,false>(a,b,empty,amax,bmax,empty,empty,out,empty,empty);
  return out;
}
torch::Tensor forward_packed(torch::Tensor a,torch::Tensor bp,torch::Tensor sf,
                             torch::Tensor amax,torch::Tensor bglobal) {
  check_common(a,bp,amax,true);check_packed(a,bp,sf,bglobal);
  auto out=torch::empty({a.size(0),bp.size(0)},a.options());
  torch::Tensor empty;
  launch<true,false>(a,bp,sf,amax,empty,bglobal,empty,out,empty,empty);
  return out;
}
std::tuple<torch::Tensor,torch::Tensor> forward_fp4(
    torch::Tensor a,torch::Tensor b,torch::Tensor amax,
    torch::Tensor bmax,torch::Tensor outmax) {
  check_common(a,b,amax,false);check_scalar(bmax,a,"B amax");
  check_scalar(outmax,a,"output amax");
  auto codes=torch::empty({a.size(0),b.size(0)/2},a.options().dtype(at::kByte));
  auto scales=torch::empty({a.size(0),b.size(0)/16},a.options().dtype(at::kByte));
  torch::Tensor empty;
  launch<false,true>(a,b,empty,amax,bmax,empty,outmax,empty,codes,scales);
  return {codes,scales};
}
std::tuple<torch::Tensor,torch::Tensor> forward_packed_fp4(
    torch::Tensor a,torch::Tensor bp,torch::Tensor sf,torch::Tensor amax,
    torch::Tensor bglobal,torch::Tensor outmax) {
  check_common(a,bp,amax,true);check_packed(a,bp,sf,bglobal);
  check_scalar(outmax,a,"output amax");
  auto codes=torch::empty({a.size(0),bp.size(0)/2},a.options().dtype(at::kByte));
  auto scales=torch::empty({a.size(0),bp.size(0)/16},a.options().dtype(at::kByte));
  torch::Tensor empty;
  launch<true,true>(a,bp,sf,amax,empty,bglobal,outmax,empty,codes,scales);
  return {codes,scales};
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_bf16",&forward_bf16,"BF16 x BF16 -> BF16, fused NVFP4 prologues");
  m.def("forward_packed",&forward_packed,"BF16 x packed NVFP4 -> BF16");
  m.def("forward_fp4",&forward_fp4,"BF16 x BF16 -> packed NVFP4");
  m.def("forward_packed_fp4",&forward_packed_fp4,"BF16 x packed NVFP4 -> packed NVFP4");
}
