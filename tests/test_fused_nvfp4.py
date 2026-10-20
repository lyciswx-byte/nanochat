"""GB10 CUDA tests; skip elsewhere. Run pytest -q tests/test_fused_nvfp4.py."""
import os
import pytest
import torch
import torch.nn.functional as F

from nanochat.kernels.fused_quant import fused_quantize, torchao_plain_reference
from nanochat.kernels.fused_rmsnorm import (
    fused_rmsnorm_quant, fused_norm_linear,
)
from nanochat.kernels.fused_mlp import fused_relu2_quant, fused_relu2_linear


def gb10():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)


pytestmark = pytest.mark.skipif(not gb10(), reason="requires CUDA 13 SM121 GB10")


@pytest.mark.parametrize("shape", [(128, 128), (128, 256), (256, 128),
                                  (256, 256), (1024, 1536)])
@pytest.mark.parametrize("kind", ["random", "zeros", "extremes"])
def test_plain_exact(shape, kind):
    torch.manual_seed(20261020)
    x = torch.randn(shape, device="cuda").bfloat16()
    if kind == "zeros":
        x.zero_()
    elif kind == "extremes":
        x[:16, :16] = 0
        x[20, 40] = 300
        x[-1, -1] = -120
    reference = torchao_plain_reference(x)
    fused = fused_quantize(x)
    for idx, (a, b) in enumerate(zip(reference, fused)):
        assert a.shape == b.shape
        assert a.dtype == b.dtype or idx == 4  # amax BF16 vs FP32 allowed
        if a.dtype == torch.float8_e4m3fn:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        assert torch.equal(a, b), f"mismatch at output {idx}, shape={shape}, kind={kind}"


@pytest.mark.parametrize("shape", [(128, 128), (128, 256), (1024, 1536)])
def test_norm_and_relu_correctness(shape):
    torch.manual_seed(6)
    x = torch.randn(shape, device="cuda").bfloat16()
    eps = float(torch.finfo(x.dtype).eps)
    expected_norm = F.rms_norm(x, (shape[-1],), eps=eps)
    expected_relu = F.relu(x).square()
    for name, fused, expected in (
        ("rmsnorm", fused_rmsnorm_quant(x, eps, True), expected_norm),
        ("relu2", fused_relu2_quant(x, True), expected_relu),
    ):
        assert torch.allclose(fused[5].float(), expected.float(), rtol=2e-2, atol=2e-2), name
        baseline = torchao_plain_reference(expected)
        a = baseline[0]
        b = fused[0]
        # Fused BF16 rounding can alter quantization near FP4 boundaries.
        # Use reference dequantized error to quantify any such differences.
        from bench_kernels import dequant
        diff = (dequant(a, baseline[1], baseline[4]) -
                dequant(b, fused[1], fused[4])).float()
        assert torch.isfinite(diff).all(), name
        assert float(diff.abs().mean()) < .035, (name, float(diff.abs().mean()))


def test_nan_visible_in_amax():
    x = torch.zeros((128, 128), dtype=torch.bfloat16, device="cuda")
    x[47, 21] = float("nan")
    # We intentionally make no assertion about codes from an invalid tensor.
    # The FP32 tensor-wide maximum must expose a NaN rather than hiding it.
    out = fused_quantize(x)
    assert torch.isnan(out[4])


def test_compile_fake_and_execute():
    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    f = torch.compile(lambda a: fused_quantize(a)[0], fullgraph=True)
    assert torch.equal(f(x), fused_quantize(x)[0])


def test_transform_autograd():
    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    eps = float(torch.finfo(x.dtype).eps)
    y = fused_rmsnorm_quant(x, eps, True)[5]
    y.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    z = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fused_relu2_quant(z, True)[5].float().square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def _make_linear():
    from nanochat.nvfp4 import convert_to_nvfp4_training
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(128, 128, bias=False, device="cuda",
                                    dtype=torch.bfloat16))
    convert_to_nvfp4_training(model)
    return model[0]


def test_fused_rmsnorm_linear_backward(monkeypatch):
    monkeypatch.setenv("NANOCHAT_NVFP4_FUSED_QUANT", "1")
    fc = _make_linear()
    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = fused_norm_linear(x, fc)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(fc.weight.grad).all()


def test_fused_relu2_linear_backward(monkeypatch):
    monkeypatch.setenv("NANOCHAT_NVFP4_FUSED_QUANT", "1")
    proj = _make_linear()
    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = fused_relu2_linear(x, proj)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(proj.weight.grad).all()
