"""Split forward/backward timing for the nanochat NVFP4 trainer."""

import os
import time

import torch

import nanochat.flash_attention as _fa

_attn = os.environ.get("ATTN", "").lower()
if _attn in ("fa2", "sdpa"):
    _fa._override_impl = _attn
    _fa.USE_FA3 = _fa._resolve_use_fa3()
    _fa.USE_FA2 = _fa._resolve_use_fa2()

import nanochat.nvfp4 as nv
from nanochat.gpt import GPT, GPTConfig
from nanochat.nvfp4 import convert_to_nvfp4_training, fuse_qkv_projections

torch.manual_seed(0)
depth = int(os.environ.get("DEPTH", "24"))
batch = int(os.environ.get("BATCH", "16"))
seq = int(os.environ.get("SEQ", "2048"))
iters = int(os.environ.get("ITERS", "6"))
mode = os.environ.get("MODE", "default")

dim = ((depth * 64 + 127) // 128) * 128
heads = dim // 128
config = GPTConfig(sequence_len=seq, vocab_size=32768, n_layer=depth,
                   n_head=heads, n_kv_head=heads, n_embd=dim,
                   window_pattern=os.environ.get("WINDOW", "SSSL"))
model = GPT(config).cuda()
model.init_weights()
if os.environ.get("MASTER", "fp32") == "bf16":
    model = model.bfloat16()
if os.environ.get("NO_CACHED_CAST", "0") == "1":
    import nanochat.nvfp4 as _nv
    _nv._install_cached_weight_cast = lambda: None
if os.environ.get("OPAQUE_RELU", "0") == "1":
    import torch.nn as nn
    from nanochat.gpt import MLP

    @torch.library.custom_op("bench::relu_square", mutates_args=())
    def _relu_square(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()

    @_relu_square.register_fake
    def _relu_square_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def _mlp_forward(self, x):
        return self.c_proj(_relu_square(self.c_fc(x)))

    MLP.forward = _mlp_forward

PREC = os.environ.get("PREC", "nvfp4")
if PREC == "nvfp4":
    fuse_qkv_projections(model)
    convert_to_nvfp4_training(model, use_rht=os.environ.get("RHT", "0") == "1")
elif PREC == "bf16":
    model = model.bfloat16()
    if os.environ.get("FUSE_QKV", "1") == "1":
        fuse_qkv_projections(model)
if os.environ.get("NO_EPI_FUSION", "0") == "1":
    import torch._inductor.config as _ic
    _ic.epilogue_fusion = False
kw = {"dynamic": False}
if mode != "default":
    kw["mode"] = mode
model = torch.compile(model, **kw)

x = torch.randint(0, 32768, (batch, seq), device="cuda")
y = torch.randint(0, 32768, (batch, seq), device="cuda")


def fwdbwd():
    model.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    return loss


for _ in range(3):
    fwdbwd()
torch.cuda.synchronize()

fwd, bwd = [], []
for _ in range(iters):
    model.zero_grad(set_to_none=True)
    s = torch.cuda.Event(enable_timing=True)
    m = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    loss = model(x, y)
    m.record()
    loss.backward()
    e.record()
    torch.cuda.synchronize()
    fwd.append(s.elapsed_time(m))
    bwd.append(m.elapsed_time(e))

fwd.sort(); bwd.sort()
print(f"peak_alloc={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
print(f"MODE={mode} DEPTH={depth} B={batch} fwd_med={fwd[len(fwd)//2]:.0f}ms "
      f"bwd_med={bwd[len(bwd)//2]:.0f}ms total={(fwd[len(fwd)//2]+bwd[len(bwd)//2]):.0f}ms",
      flush=True)
