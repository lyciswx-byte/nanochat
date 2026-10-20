"""GB10-only correctness/opacity/regression tests for triton_fused.py.

Run: NANOCHAT_TRITON_FUSED=1 pytest -q tests/test_triton_fused.py
Large matrix checks: ... pytest -q -m slow tests/test_triton_fused.py
No GPU is required merely to collect this file; tests skip without SM121.
"""
import os

import pytest
import torch


@pytest.fixture(autouse=True)
def enable_kernels(monkeypatch):
    monkeypatch.setenv("NANOCHAT_TRITON_FUSED", "1")


@pytest.fixture(scope="module")
def gb10():
    pytest.importorskip("triton")
    pytest.importorskip("torchao")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("tests require DGX Spark sm_121a")
    return torch.device("cuda")


def _rand(m, k, device, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((m, k), device=device, generator=g, dtype=torch.bfloat16)


@pytest.mark.parametrize("shape", [(128, 128), (256, 256), (256, 384)])
def test_exact_torchao_same_amax(shape, gb10):
    from nanochat.kernels.triton_fused import quantize_2d_static
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )
    x = _rand(*shape, gb10)
    amax = x.abs().max().float()
    reference = triton_weight_quantize_2d(x, amax)
    candidate = quantize_2d_static(x, amax)
    for index, (actual, expected) in enumerate(zip(candidate, reference)):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected), f"layout/code mismatch: output {index}"


def test_zero_tensors_zero_scale_and_fresh_outputs(gb10):
    from nanochat.kernels.triton_fused import quantize_2d_static
    x = torch.zeros((128, 128), device=gb10, dtype=torch.bfloat16)
    amax = torch.zeros((), device=gb10, dtype=torch.float32)
    first, second = quantize_2d_static(x, amax), quantize_2d_static(x, amax)
    for a, b in zip(first, second):
        assert a.data_ptr() != b.data_ptr(), "returned aliased/persistent tensor!"
        assert torch.equal(a, b)
        assert torch.count_nonzero(a.float()) == 0


def test_col_transpose_independently_quantized(gb10):
    from nanochat.kernels.triton_fused import quantize_2d_static
    x = _rand(256, 384, gb10)
    amax = x.abs().max().float()
    row, sf, col, colsf = quantize_2d_static(x, amax)
    trow, tsf, _, _ = quantize_2d_static(x.T.contiguous(), amax)
    assert torch.equal(col, trow), "column codes must equal W.T row codes"
    assert torch.equal(colsf, tsf), "column scales must equal W.T scales"
    assert row.shape == (256, 192) and sf.shape == (2, 6, 32, 16)


def test_static_error_vs_dynamic(gb10):
    from nanochat.kernels.triton_fused import quantize_2d_static
    from bench_triton_fused import dequantize
    x = _rand(256, 384, gb10)
    dynamic = x.abs().max().float()
    static = torch.tensor(6.0, device=gb10)
    packed = quantize_2d_static(x, static)
    row, col = dequantize(packed[0], packed[1], static), dequantize(packed[2], packed[3], static)
    assert torch.isfinite(row.float()).all()
    assert torch.isfinite(col.float()).all()
    assert ((row.float() - x.float()).square().mean().sqrt() < 0.20)
    assert ((col.float() - x.T.float()).square().mean().sqrt() < 0.20)
    reference = quantize_2d_static(x, dynamic)
    assert reference[0].shape == packed[0].shape


@pytest.mark.parametrize("k", [1536, 6144])
def test_residual_rmsnorm_fwd_and_explicit_backward(gb10, k):
    from nanochat.kernels.triton_fused import (
        residual_rmsnorm_quantize, rmsnorm_residual_backward,
        quantize_2d_static,
    )
    import torch.nn.functional as F
    x = _rand(128, k, gb10, 77)
    residual = _rand(128, k, gb10, 88)
    amax = torch.tensor(6.0, device=gb10, dtype=torch.float32)
    packed = residual_rmsnorm_quantize(x, residual, amax)
    row, sf, col, colsf, rstd = packed
    z = (x.float() + residual.float()).to(torch.bfloat16)
    norm = F.rms_norm(z, (k,), eps=torch.finfo(torch.bfloat16).eps)
    expected = quantize_2d_static(norm.contiguous(), amax)
    # Bound ties can differ at fp4 midpoints due to rsqrt versus native RMSNorm
    # rounding; check scales and decoded outputs rather than only byte identity.
    from bench_triton_fused import dequantize
    got_deq = dequantize(row, sf, amax)
    ref_deq = dequantize(expected[0], expected[1], amax)
    assert (got_deq.float() - ref_deq.float()).square().mean().sqrt() < 0.08
    expected_rstd = (z.float().square().mean(dim=-1) +
                     torch.finfo(torch.bfloat16).eps).rsqrt()
    torch.testing.assert_close(rstd, expected_rstd, atol=1e-3, rtol=1e-3)
    assert row.shape == (128, k // 2)
    assert col.shape == (k, 64)
    assert colsf.shape == (k // 128, 2, 32, 16)

    dy = _rand(128, k, gb10, 123)
    dx, dr = rmsnorm_residual_backward(x, residual, dy, rstd)
    dot = (z.float() * dy.float()).mean(dim=1, keepdim=True)
    expected_grad = (rstd[:, None] * dy.float()
                     - z.float() * (rstd[:, None] ** 3) * dot).bfloat16()
    torch.testing.assert_close(dx, expected_grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dr, expected_grad, atol=2e-2, rtol=2e-2)


def test_rstd_autograd(gb10):
    from nanochat.kernels.triton_fused import residual_rmsnorm_quantize
    x = _rand(128, 1536, gb10).requires_grad_(True)
    residual = _rand(128, 1536, gb10, 123).requires_grad_(True)
    _, _, _, _, rstd = residual_rmsnorm_quantize(
        x, residual, torch.tensor(6.0, device=gb10)
    )
    rstd.sum().backward()
    z = (x.detach().float() + residual.detach().float()).to(torch.bfloat16).float()
    expected = -z * (rstd.detach()[:, None] ** 3) / z.shape[-1]
    torch.testing.assert_close(x.grad, expected.bfloat16(), atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(residual.grad, expected.bfloat16(), atol=1e-3, rtol=1e-2)


def test_fake_outputs_and_compile_no_persistent_donation(gb10):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from nanochat.kernels.triton_fused import quantize_2d_static, residual_rmsnorm_quantize
    x = _rand(128, 128, gb10)
    amax = torch.tensor(6.0, device=gb10)
    mode = FakeTensorMode()
    with mode:
        fake_x = mode.from_tensor(x)
        fake_amax = mode.from_tensor(amax)
        codes, sf, tcodes, tsf = quantize_2d_static(fake_x, fake_amax)
        assert codes.shape == (128, 64) and codes.dtype == torch.uint8
        assert sf.shape == (1, 2, 32, 16)
        assert tcodes.shape == (128, 64)
        assert tsf.shape == (1, 2, 32, 16)
        rms = residual_rmsnorm_quantize(fake_x, fake_x, fake_amax)
        assert rms[-1].shape == (128,) and rms[-1].dtype == torch.float32

    compiled = torch.compile(
        lambda y: quantize_2d_static(y, amax), backend="inductor"
    )
    one = compiled(x)
    two = compiled(x)
    for a, b in zip(one, two):
        assert torch.equal(a, b)
        assert a.data_ptr() != b.data_ptr(), "Inductor reused live custom-op output"


def test_env_default_off(gb10, monkeypatch):
    from nanochat.kernels.triton_fused import quantize_2d_static
    monkeypatch.delenv("NANOCHAT_TRITON_FUSED", raising=False)
    with pytest.raises(RuntimeError, match="NANOCHAT_TRITON_FUSED"):
        quantize_2d_static(_rand(128, 128, gb10), 6.0)


@pytest.mark.slow
@pytest.mark.parametrize("shape", [(32768, 1536), (32768, 6144)])
def test_d24_shape_reconstruction(gb10, shape):
    from nanochat.kernels.triton_fused import quantize_2d_static
    from bench_triton_fused import dequantize
    x = _rand(*shape, gb10)
    amax = torch.tensor(6.0, device=gb10)
    code, sf, code_t, sf_t = quantize_2d_static(x, amax)
    row = dequantize(code, sf, amax)
    col = dequantize(code_t, sf_t, amax)
    assert torch.isfinite(row.float()).all()
    assert torch.isfinite(col.float()).all()
    assert (row.float() - x.float()).square().mean().sqrt() < 0.2
    assert (col.float() - x.T.float()).square().mean().sqrt() < 0.2
