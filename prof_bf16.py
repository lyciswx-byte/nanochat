"""Profile the BF16 (BF16 master) d24 training step."""

import os

import torch

from nanochat.gpt import GPT, GPTConfig

torch.manual_seed(0)
depth = int(os.environ.get("DEPTH", "24"))
batch = int(os.environ.get("BATCH", "16"))
dim = ((depth * 64 + 127) // 128) * 128
heads = dim // 128
cfg = GPTConfig(sequence_len=2048, vocab_size=32768, n_layer=depth,
                n_head=heads, n_kv_head=heads, n_embd=dim, window_pattern="SSSL")
model = GPT(cfg).cuda()
model.init_weights()
model = model.bfloat16()
model = torch.compile(model, dynamic=False)
x = torch.randint(0, 32768, (batch, 2048), device="cuda")
y = torch.randint(0, 32768, (batch, 2048), device="cuda")


def step():
    model.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()


for _ in range(3):
    step()
torch.cuda.synchronize()

from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
prof.export_chrome_trace("/workspace/nanochat/prof_bf16.json")
print("exported", flush=True)
