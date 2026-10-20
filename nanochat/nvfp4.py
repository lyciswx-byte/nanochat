"""NVFP4 training support for GB10 (sm_121a), built on torchao.

torchao's NVFP4 training linear quantizes all three GEMMs of a Linear layer
(forward, dgrad, wgrad) to block-scaled FP4 and runs them on Blackwell's
native packed ``mxf4nvf4`` tensor-core instruction. On GB10 that is roughly
2x the FP8 GEMM throughput.

The one incompatibility: torchao's backward quantization uses stochastic
rounding (``cvt.rs``), which ptxas rejects for ``sm_121a`` ("Feature '.rs'
not supported on .target 'sm_121a'"). We force round-to-nearest for that
quantization instead; the GEMM path and the block scales are unchanged.
"""

import torch
import torch.nn as nn
import inspect
import os
import types

from nanochat.gpt import FusedQKVLinear

from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
    _rht_quantize_row_col as _orig_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    NVFP4Linear,
    NVFP4TrainingConfig,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization import quantize_


def _disable_stochastic_rounding():
    """Route torchao's backward NR quantize through RTNE (sm_121a has no cvt.rs)."""

    def _rht_quantize_row_col_rtne(*args, **kwargs):
        kwargs["sr_seed"] = None
        return _orig_rht_quantize_row_col(*args, **kwargs)

    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nvfp4_linear

    if not getattr(nvfp4_linear, "_nanochat_rtne_patched", False):
        nvfp4_linear._rht_quantize_row_col = _rht_quantize_row_col_rtne
        nvfp4_linear._nanochat_rtne_patched = True


def _install_fast_weight_quantizer():
    """Drop the two ``torch.zeros`` calls around torchao's 2D quantize kernel.

    The kernel stores every output tile through TMA, so the zero-fill is dead
    work (5-8% of a micro-batch) whenever M and N are exact multiples of the
    largest autotuned tile (256).
    """
    import torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton as q2

    if getattr(q2, "_nanochat_no_fill", False):
        return

    def _quantize_2d_no_fill_impl(
        A: torch.Tensor, global_amax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        M, N = A.shape
        aligned = M % 256 == 0 and N % 256 == 0
        if aligned:
            a_fp4 = A.new_empty((M, N // 2), dtype=torch.uint8)
            a_t_fp4 = A.new_empty((N, M // 2), dtype=torch.uint8)
        else:
            a_fp4 = torch.zeros((M, N // 2), dtype=torch.uint8, device=A.device)
            a_t_fp4 = torch.zeros((N, M // 2), dtype=torch.uint8, device=A.device)
        a_sf = A.new_empty((M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
        a_t_sf = A.new_empty((N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn)

        import triton
        from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
            prepare_for_cuda_graph,
        )

        num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
        try:
            if hasattr(triton, "set_allocator"):
                _ws = prepare_for_cuda_graph(A.device)
                triton.set_allocator(
                    lambda size, align, stream: _ws[: max(size, 1)]
                )
            q2.triton_quantize_2d_weight[(num_sms,)](
                A, a_fp4, a_sf, a_t_fp4, a_t_sf, global_amax, M, N,
                GROUP_SIZE_N=8, NUM_SMS=num_sms,
            )
        finally:
            if hasattr(triton, "set_allocator"):
                triton.set_allocator(None)
        return a_fp4, a_sf, a_t_fp4, a_t_sf

    # Must be an opaque custom op: torch.compile traces into the autograd
    # Function's forward with FakeTensors, and a bare Triton kernel cannot run
    # there. The fake impl below mirrors torchao's.
    _quantize_2d_no_fill = torch.library.custom_op(
        "nanochat::nvfp4_quantize_2d_nofill", _quantize_2d_no_fill_impl, mutates_args=()
    )

    @_quantize_2d_no_fill.register_fake
    def _(
        A: torch.Tensor, global_amax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        M, N = A.shape
        return (
            A.new_empty((M, N // 2), dtype=torch.uint8),
            A.new_empty((M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn),
            A.new_empty((N, M // 2), dtype=torch.uint8),
            A.new_empty((N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn),
        )

    q2.triton_weight_quantize_2d = _quantize_2d_no_fill
    q2._nanochat_no_fill = True
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nvfp4_linear
    nvfp4_linear.triton_weight_quantize_2d = _quantize_2d_no_fill


_wq_cache: dict = {}
_wq_stats = {"hit": 0, "miss": 0}


def _install_weight_quant_cache():
    """Reuse the quantized weight across gradient-accumulation micro-steps.

    Weights only change in ``optimizer.step()`` (all in-place, so ``_version``
    bumps). The forward otherwise re-quantizes the same 730M weights once per
    micro-batch, which costs a quantize kernel + a global amax per Linear.
    """
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nvfp4_linear

    if getattr(nvfp4_linear, "_nanochat_weight_cache", False):
        return
    orig = nvfp4_linear._weight_quantize_2d

    def _cached_impl(
        x: torch.Tensor, use_cutedsl: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # torch.compile hands us functionalized views of the parameter, so the
        # Python object identity changes every call; the storage pointer and
        # the version counter do not.
        key = (x.data_ptr(), x._version, tuple(x.shape))
        hit = _wq_cache.get(key)
        if hit is not None:
            _wq_stats["hit"] += 1
        else:
            _wq_stats["miss"] += 1
            hit = orig(x, use_cutedsl)
            if len(_wq_cache) >= 1024:
                _wq_cache.clear()
            _wq_cache[key] = hit
        # Inductor owns custom-op outputs and may donate/free their storage.
        # Returning the cached tensors themselves lets it overwrite the packed
        # weights (verified corruption: loss 10.40 -> 24.81 on replay). Hand
        # back fresh allocations; cloning ~0.7 GB per micro-step is far cheaper
        # than re-running the quantizer over every weight.
        return tuple(t.clone() for t in hit)

    # Opaque custom op: without this, AOTAutograd traces the autograd.Function
    # and inlines the quantize into the compiled graph, so the cache check never
    # runs at replay time.
    _cached = torch.library.custom_op(
        "nanochat::weight_quantize_2d_cached", _cached_impl, mutates_args=()
    )

    @_cached.register_fake
    def _(
        x: torch.Tensor, use_cutedsl: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return orig(x, use_cutedsl)

    nvfp4_linear._weight_quantize_2d = _cached
    nvfp4_linear._nanochat_weight_cache = True


# ---------------------------------------------------------------------------
# Stable BF16 shadow buffer for the FP32 master weight
# ---------------------------------------------------------------------------
# ``nvfp4_mm_triton.forward`` casts a non-BF16 weight with ``weight.to(bf16)``.
# Training keeps FP32 master weights, so that cast materialises a fresh tensor
# on every micro-step and the cached weight quantizer never hits. A registered
# non-persistent BF16 buffer is a stable graph input: the quantizer sees the
# same storage across micro-steps, while ``refresh_nvfp4_weight_shadows`` copies
# the new master value once after each optimizer update. The buffer is never
# returned as an ephemeral cache entry, so Inductor cannot reuse its storage.
_W_SHADOW = "_nanochat_bf16_weight"


class _ShadowBf16Weight(torch.autograd.Function):
    """Forward uses the stable BF16 shadow; backward returns FP32 master grads."""

    @staticmethod
    def forward(ctx, master: torch.Tensor, shadow: torch.Tensor):
        ctx.master_dtype = master.dtype
        return shadow

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad.to(ctx.master_dtype), None


def _install_weight_shadow_forward():
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_training as nvfp4_training
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
        nvfp4_linear as _nvfp4_linear_fn,
    )

    cls = nvfp4_training.NVFP4Linear
    if getattr(cls, "_nanochat_shadow_forward", False):
        return
    orig_forward = cls.forward

    def forward(self, x):
        shadow = getattr(self, _W_SHADOW, None)
        if shadow is None or self.process_group is not None:
            return orig_forward(self, x)
        return _nvfp4_linear_fn(
            x,
            _ShadowBf16Weight.apply(self.weight, shadow),
            self.bias,
            kernel_preference=self.kernel_preference,
            sr_seed=self._sr_seed,
            sign_vector=self.rht_sign_vector,
        )

    cls.forward = forward
    cls._nanochat_shadow_forward = True


@torch.no_grad()
def refresh_nvfp4_weight_shadows(model: nn.Module) -> None:
    """Copy the FP32 masters into their BF16 shadows after an optimizer step."""
    for module in model.modules():
        shadow = getattr(module, _W_SHADOW, None)
        if shadow is not None:
            shadow.copy_(module.weight)


def invalidate_weight_cache():
    _wq_cache.clear()


def _use_plain_quantization():
    """Replace torchao's RHT activation quantizer with its plain 2D block quantizer.

    RHT (a Hadamard rotation before quantization) improves accuracy but costs a
    separate amax pass plus a transform over every activation tensor, which on
    GB10's 273 GB/s memory eats the FP4 GEMM speedup. The plain quantizer uses
    the same 16-element block scales and the same FP4 GEMM, just without the
    rotation.
    """
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nvfp4_linear
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        triton_weight_quantize_2d,
    )

    def _plain_amax(x, sign_vector_list, use_cutedsl):
        # The input is BF16 and the NVFP4 block scale is E4M3, so converting
        # every activation tile to FP32 just to find a global maximum adds a
        # full tensor pass without changing the scale at useful precision.
        # Keep an opt-out for precision experiments; BF16 reduction is the
        # fast default on GB10.
        if use_cutedsl or os.environ.get("NANOCHAT_NVFP4_AMAX_FP32", "0") == "1":
            amax = x.float().abs().max()
        else:
            # ``x.abs().amax()`` materialises a full-size absolute-value tensor
            # in eager mode (three passes over the activation). The infinity
            # norm is a single fused reduction and measures ~3x faster for the
            # same value; torch.compile keeps it as one reduction kernel.
            amax = torch.linalg.vector_norm(x, ord=float("inf"))
        return amax, amax

    def _plain_quantize(
        x,
        col_amax,
        row_amax,
        sign_vector_list,
        use_cutedsl,
        use_fast_math=True,
        sr_seed=None,
        **kwargs,
    ):
        """Quantize activations without the RHT transform.

        torchao 0.19 added ``use_fast_math`` before ``sr_seed``.  Accepting
        both the old and new call forms keeps the plain path usable across
        the CUDA images used for the Spark pair; ``sr_seed`` is intentionally
        ignored because this path uses deterministic RTNE quantization.
        """
        if use_cutedsl:
            from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
                cutedsl_weight_quantize_2d,
            )
            row_codes, row_sf, col_codes, col_sf = cutedsl_weight_quantize_2d(
                x, row_amax
            )
        else:
            row_codes, row_sf, col_codes, col_sf = triton_weight_quantize_2d(
                x, row_amax
            )
        return col_codes, col_sf, row_codes, row_sf

    nvfp4_linear._rht_amax = _plain_amax
    nvfp4_linear._rht_quantize_row_col = _plain_quantize


def _install_quack_scaled_mm():
    """Use Quack's SM120/121 block-scaled MMA for torchao's three GEMMs.

    torchao already produces the canonical NVFP4 storage used by Quack: packed
    E2M1 values, 16-element E4M3 scale blocks, and a scalar tensor scale.  The
    only representation difference is that torchao keeps the scale atom as a
    four-dimensional ``(..., 32, 16)`` tensor while Quack accepts the equivalent
    ``(..., 32, 4, 4)`` view.  Keeping the wrapper local to torchao's module
    avoids changing ``torch.nn.functional.scaled_mm`` for unrelated callers.
    """
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nvfp4_linear

    if getattr(nvfp4_linear, "_nanochat_quack_scaled_mm", False):
        return
    try:
        from quack.blockscaled import BlockScaledOperand, NVFP4
        from quack.gemm_interface import gemm as quack_gemm
    except ImportError:
        # The native CUDA container ships Quack; source-only environments do
        # not.  Keep torchao's scaled_mm path available in the latter.
        return

    torch_f = nvfp4_linear.F
    orig_scaled_mm = torch_f.scaled_mm

    def _reshape_sf(sf: torch.Tensor, rows: int, logical_k: int) -> torch.Tensor:
        # torchao's flattened scale output is contiguous in the same order as
        # Quack's 128x4 blocked atom.  Avoid a copy; this is on every GEMM.
        expected = (rows // 128, logical_k // 64, 32, 4, 4)
        return sf.view(expected)

    def _quack_scaled_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_recipe_a,
        scale_b,
        scale_recipe_b,
        swizzle_a=None,
        swizzle_b=None,
        bias=None,
        output_dtype=torch.bfloat16,
        contraction_dim=(),
        use_fast_accum=False,
    ):
        # Quack implements the exact NVFP4 recipe used by this module.  For a
        # future torchao recipe, delegate instead of silently changing math.
        if (
            not isinstance(scale_a, (list, tuple))
            or not isinstance(scale_b, (list, tuple))
            or len(scale_a) != 2
            or len(scale_b) != 2
            or mat_a.dtype != torch.float4_e2m1fn_x2
            or mat_b.dtype != torch.float4_e2m1fn_x2
        ):
            return orig_scaled_mm(
                mat_a,
                mat_b,
                scale_a=scale_a,
                scale_recipe_a=scale_recipe_a,
                scale_b=scale_b,
                scale_recipe_b=scale_recipe_b,
                swizzle_a=swizzle_a,
                swizzle_b=swizzle_b,
                bias=bias,
                output_dtype=output_dtype,
                contraction_dim=contraction_dim,
                use_fast_accum=use_fast_accum,
            )
        m = mat_a.shape[-2]
        logical_k = mat_a.shape[-1] * 2
        n = mat_b.shape[-1]
        # B is passed to scaled_mm as (K/2, N), whereas BlockScaledOperand's
        # canonical quantization view is (N, K/2) followed by .mT.
        a = BlockScaledOperand.from_parts(
            mat_a,
            _reshape_sf(scale_a[0], m, logical_k),
            NVFP4,
            per_tensor_scale=scale_a[1].reshape(1),
        )
        b = BlockScaledOperand.from_parts(
            mat_b.transpose(-1, -2),
            _reshape_sf(scale_b[0], n, logical_k),
            NVFP4,
            per_tensor_scale=scale_b[1].reshape(1),
        ).mT
        return quack_gemm(a, b, bias=bias, out_dtype=output_dtype)

    # nvfp4_linear only reads F.ScalingType/SwizzleType and F.scaled_mm.
    # Use a small proxy rather than mutating the process-wide functional module.
    nvfp4_linear.F = types.SimpleNamespace(
        ScalingType=torch_f.ScalingType,
        SwizzleType=torch_f.SwizzleType,
        scaled_mm=_quack_scaled_mm,
    )
    nvfp4_linear._nanochat_quack_scaled_mm = True


def _install_fast_accum_scaled_mm():
    """Opt into the hardware fast-accumulate NVFP4 MMA variant.

    GB10 exposes this through ``scaled_mm(..., use_fast_accum=True)``. It
    trades a small amount of accumulator precision for lower tensor-core
    latency and is kept behind an environment switch until the precision
    comparison has been run.
    """
    import torchao.prototype.moe_training.nvfp4_training.nvfp4_linear as nvfp4_linear

    if getattr(nvfp4_linear, "_nanochat_fast_accum", False):
        return
    orig_f = nvfp4_linear.F
    orig_scaled_mm = orig_f.scaled_mm

    class _FProxy:
        def __getattr__(self, name):
            return getattr(orig_f, name)

        def scaled_mm(self, *args, **kwargs):
            kwargs["use_fast_accum"] = True
            return orig_scaled_mm(*args, **kwargs)

    nvfp4_linear.F = _FProxy()
    nvfp4_linear._nanochat_fast_accum = True


def nvfp4_module_filter(mod: nn.Module, fqn: str) -> bool:
    """Only dense Linears whose dims satisfy torchao's FP4 tiling constraints."""
    if not isinstance(mod, nn.Linear) or isinstance(mod, NVFP4Linear):
        return False
    return mod.in_features % 128 == 0 and mod.out_features % 128 == 0


def fuse_qkv_projections(model: nn.Module) -> int:
    """Fuse each attention Q/K/V projection into one equivalent Linear.

    This preserves the number of parameters and the Q/K/V split exactly while
    allowing NVFP4 to quantize and launch one larger GEMM per attention layer.
    It must run after checkpoint loading and before ``quantize_``.
    """
    fused = 0
    layers = getattr(getattr(model, "transformer", None), "h", ())
    for block in layers:
        attn = getattr(block, "attn", None)
        if attn is None or getattr(attn, "c_qkv", None) is not None:
            continue
        q, k, v = attn.c_q, attn.c_k, attn.c_v
        attn.c_qkv = FusedQKVLinear.from_qkv(q, k, v)
        del attn.c_q
        del attn.c_k
        del attn.c_v
        fused += 1
    return fused


def convert_to_nvfp4_training(
    model: nn.Module, module_filter_fn=nvfp4_module_filter, use_rht: bool = False
) -> int:
    """Replace eligible nn.Linear modules with NVFP4Linear. Returns the count.

    use_rht=True keeps torchao's randomized-Hadamard activation quantizer
    (more accurate, noticeably slower). Default False uses the plain
    16-element block quantizer.
    """
    _disable_stochastic_rounding()
    _install_fast_weight_quantizer()
    if not use_rht:
        _use_plain_quantization()
    # The cached quantizer returns clones of the packed weights (see
    # ``_cached_impl``), so Inductor owns every tensor it may donate.
    if os.environ.get("NANOCHAT_NVFP4_CACHE_WEIGHTS", "1") == "1":
        _install_weight_quant_cache()
    shadow_enabled = os.environ.get("NANOCHAT_NVFP4_WEIGHT_SHADOW", "1") == "1"
    if shadow_enabled:
        _install_weight_shadow_forward()
    if os.environ.get("NANOCHAT_NVFP4_FAST_ACCUM", "0") == "1":
        _install_fast_accum_scaled_mm()
    if os.environ.get("NANOCHAT_NVFP4_GEMM", "torchao").lower() == "quack":
        _install_quack_scaled_mm()
    # CuteDSL's default AUTO selection targets datacenter Blackwell tcgen05
    # (sm_100/103). GB10 is consumer SM121 and must use the warp-level Triton
    # quantizers; leaving AUTO enabled can fail during compile with a misleading
    # arch error before any useful kernel is built.
    # ``use_fast_math`` was added after the torchao build shipped in the
    # GB10 CUDA container.  Keep the explicit Triton backend on both APIs,
    # but only pass the optional flag when this torchao version supports it.
    kernel_name = os.environ.get("NANOCHAT_NVFP4_KERNEL", "triton").lower()
    if kernel_name not in {"triton", "cutedsl"}:
        raise ValueError("NANOCHAT_NVFP4_KERNEL must be 'triton' or 'cutedsl'")
    kernel_preference = (
        KernelPreference.CUTEDSL
        if kernel_name == "cutedsl"
        else KernelPreference.TRITON
    )
    config_kwargs = {"kernel_preference": kernel_preference}
    try:
        if "use_fast_math" in inspect.signature(NVFP4TrainingConfig).parameters:
            config_kwargs["use_fast_math"] = True
    except (TypeError, ValueError):
        pass
    nvfp4_config = NVFP4TrainingConfig(**config_kwargs)
    quantize_(model, nvfp4_config, filter_fn=module_filter_fn)
    # torchao's Triton quantizer keeps a persistent TMA workspace in a module
    # global. Allocate it before torch.compile so CUDA-Graph capture does not
    # create a live tensor in its private pool and then retain it globally.
    try:
        from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
            prepare_for_cuda_graph,
        )
        device = next(model.parameters()).device
        if device.type == "cuda":
            prepare_for_cuda_graph(device)
    except (ImportError, StopIteration, AttributeError):
        # Older torchao wheels do not expose the prewarm helper; eager mode
        # remains usable there and the compile-thread guard still applies.
        pass
    n = 0
    for mod in model.modules():
        if isinstance(mod, NVFP4Linear):
            # Keep checkpoints byte-compatible with a plain GPT state_dict:
            # the RHT seed/basis are run-local, not model weights.
            mod._non_persistent_buffers_set.update({"_sr_seed", "_rht_sign_vector"})
            if shadow_enabled and mod.process_group is None and mod.weight.dtype != torch.bfloat16:
                if getattr(mod, _W_SHADOW, None) is None:
                    mod.register_buffer(
                        _W_SHADOW,
                        mod.weight.detach().to(torch.bfloat16).clone(),
                        persistent=False,
                    )
            n += 1
    return n
