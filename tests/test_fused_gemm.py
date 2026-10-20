"""sm_121a fused GEMM tests.

CUDA tests require:
  NANOCHAT_FUSED_GEMM_ENABLE=1 pytest -q tests/test_fused_gemm.py

Tests intentionally do not run on non-GB10 hardware and never claim that
a compile-only / FakeTensor pass validates PTX numeric correctness.
"""
from __future__ import annotations

import os

import pytest
import torch

from nanochat.kernels.fused_gemm import PackedNVFP4, dequantize, fused_gemm


def _has_gb10() -> bool:
    return (torch.cuda.is_available() and
            torch.cuda.get_device_capability() == (12, 1) and
            os.environ.get("NANOCHAT_FUSED_GEMM_ENABLE") == "1")


gb10 = pytest.mark.skipif(not _has_gb10(),
                          reason="opt-in GB10 sm_121a GPU required")


def _data(m=64,n=64,k=64, seed=7):
    torch.manual_seed(seed)
    a=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
    b=torch.randn(n,k,device="cuda",dtype=torch.bfloat16)
    return a,b


def _nrmse(x,y):
    return float(((x.float()-y.float()).square().mean().sqrt() /
                  y.float().square().mean().sqrt().clamp_min(1e-8)).item())


def _pack_reference(b: torch.Tensor) -> PackedNVFP4:
    """Independent tensorwise+blockwise E4M3 / E2M1 reference packer."""
    n,k=b.shape
    global_scale=b.float().abs().max()/2688.0
    blocks=b.float().reshape(n,k//16,16)
    bs=(blocks.abs().amax(dim=-1)/6/global_scale.clamp_min(1e-30))
    bs=bs.clamp(0,448).to(torch.float8_e4m3fn)
    scale=bs.float()*global_scale
    q=blocks / scale.clamp_min(1e-30).unsqueeze(-1)
    codebook=torch.tensor([0,.5,1,1.5,2,3,4,6],device=b.device)
    # RNE tie-handling: representable-index parity breaks equal-distance ties.
    nearest=(q.abs().unsqueeze(-1)-codebook).abs()
    nearest=nearest+torch.arange(8,device=b.device).float().remainder(2)*1e-6
    mag=nearest.argmin(dim=-1).to(torch.uint8)
    neg=(q<0).to(torch.uint8)*8
    code=(mag|neg).reshape(n,k)
    packed=(code[:,0::2] | (code[:,1::2]<<4)).contiguous()
    return PackedNVFP4(packed,bs.view(torch.uint8).contiguous(),
                       global_scale.reshape(1).float())


def test_default_off(monkeypatch):
    monkeypatch.delenv("NANOCHAT_FUSED_GEMM_ENABLE",raising=False)
    a=torch.zeros((64,64),dtype=torch.bfloat16)
    with pytest.raises(RuntimeError,match="ENABLE"):
        fused_gemm(a,a)


def test_fake_opaque_ops():
    """Faithful fake shape/dtype, with no CUDA toolchain or GPU needed."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        a=torch.empty(64,128,device="cpu",dtype=torch.bfloat16)
        b=torch.empty(128,128,device="cpu",dtype=torch.bfloat16)
        scal=torch.empty(1,device="cpu",dtype=torch.float32)
        y=torch.ops.nanochat.sm121_fused_gemm_bf16(a,b,scal,scal)
        assert y.shape==(64,128) and y.dtype==torch.bfloat16
        bp=torch.empty((128,64),device="cpu",dtype=torch.uint8)
        sf=torch.empty((128,8),device="cpu",dtype=torch.uint8)
        yp=torch.ops.nanochat.sm121_fused_gemm_packed(a,bp,sf,scal,scal)
        assert yp.shape==(64,128)
        cp,sp=torch.ops.nanochat.sm121_fused_gemm_out_fp4(a,b,scal,scal,scal)
        assert (cp.shape,sp.shape)==((64,64),(64,8))
        cp,sp=torch.ops.nanochat.sm121_fused_gemm_packed_out_fp4(a,bp,sf,scal,scal,scal)
        assert (cp.shape,sp.shape)==((64,64),(64,8))
        assert torch.ops.nanochat.sm121_fused_gemm_grad_a(y,b).shape==(64,128)
        assert torch.ops.nanochat.sm121_fused_gemm_grad_b(y,a).shape==(128,128)


@gb10
@pytest.mark.parametrize("m,n,k",[(64,64,64),(128,128,128),(128,64,256)])
def test_forward_vs_bf16(m,n,k):
    a,b=_data(m,n,k)
    y=fused_gemm(a,b)
    ref=a.float() @ b.float().T
    assert y.shape==(m,n) and y.dtype==torch.bfloat16
    assert torch.isfinite(y).all()
    # The FP4 quantization error is expected; guard egregious fragment errors.
    assert _nrmse(y,ref)<0.25


@gb10
def test_static_equal_amax_matches_exact():
    a,b=_data(128,64,128)
    exact=fused_gemm(a,b,scale_mode="exact")
    sa=a.float().abs().max().reshape(1)
    sb=b.float().abs().max().reshape(1)
    static=fused_gemm(a,b,scale_mode="static",
                      static_amax_a=sa,static_amax_b=sb)
    torch.testing.assert_close(static,exact,atol=0,rtol=0)
    # Outputs from successive custom op calls must never alias or be cached.
    assert exact.data_ptr()!=static.data_ptr()


@gb10
def test_output_fp4_dequantized_and_layout():
    a,b=_data(64,128,128)
    exact=fused_gemm(a,b)
    p=fused_gemm(a,b,out_fp4=True,
                 out_static_amax=exact.float().abs().max()*1.25)
    assert isinstance(p,PackedNVFP4)
    assert p.codes.shape==(64,64) and p.scales.shape==(64,8)
    assert p.codes.dtype==torch.uint8 and p.scales.dtype==torch.uint8
    reconstructed=dequantize(p)
    assert reconstructed.shape==(64,128)
    assert torch.isfinite(reconstructed).all()
    assert _nrmse(reconstructed,exact)<0.30


@gb10
def test_packed_b_equivalent_to_quantizing_b_in_prologue():
    a,b=_data(128,64,128)
    ref=_pack_reference(b)
    # Two paths should have equivalent E2M1 / E4M3 values up to tie cases.
    y1=fused_gemm(a,b)
    y2=fused_gemm(a,packed_b=ref)
    assert _nrmse(y2,y1)<0.08


@gb10
def test_gradients_opaque_and_ste():
    a,b=_data(64,128,128)
    a.requires_grad_()
    b.requires_grad_()
    y=fused_gemm(a,b,scale_mode="exact")
    g=y.float().square().mean()
    g.backward()
    dy=(2*y.float()/y.numel()).to(torch.bfloat16)
    # Explicit straight-through derivative w.r.t. BF16 inputs.
    ea=(dy @ b).to(a.dtype)
    eb=(dy.T @ a).to(b.dtype)
    assert a.grad is not None and b.grad is not None
    torch.testing.assert_close(a.grad,ea,atol=0,rtol=0)
    torch.testing.assert_close(b.grad,eb,atol=0,rtol=0)


@gb10
def test_compiled_replay_does_not_alias_outputs():
    a,b=_data(64,64,64)
    sa=a.float().abs().max().reshape(1)
    sb=b.float().abs().max().reshape(1)
    def f(x,w):
        return fused_gemm(x,w,scale_mode="static",
                          static_amax_a=sa,static_amax_b=sb)
    compiled=torch.compile(f,fullgraph=True)
    first=compiled(a,b)
    second=compiled(a,b)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    # In-place modification of first cannot corrupt output of second.
    first.fill_(float("nan"))
    assert torch.isfinite(second).all()
