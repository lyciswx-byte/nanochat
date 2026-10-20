"""Opt-in chunked LM-head + softcapped cross-entropy.

The reference GPT materializes [tokens, vocab] BF16 logits and several FP32
copies. This opaque forward/backward pair keeps only one row-chunk live at a
time. It deliberately preserves the original BF16 Linear matmul and the FP32
softcap/CE ordering; use NANOCHAT_CHUNKED_CE=1 to enable in training.
"""
import torch
import torch.nn.functional as F


@torch.library.custom_op("nanochat::chunked_lm_ce", mutates_args=())
def _chunked_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    vocab: int,
    rows: int,
    reduction: str,
) -> torch.Tensor:
    tokens = x.reshape(-1, x.shape[-1])
    labels = targets.reshape(-1)
    # GPT.Linear explicitly casts the FP32 master to the activation dtype.
    w = weight[:vocab].to(tokens.dtype)
    pieces = []
    total = torch.zeros((), dtype=torch.float32, device=x.device)
    with torch.no_grad():
        for start in range(0, tokens.shape[0], rows):
            end = min(start + rows, tokens.shape[0])
            logits = F.linear(tokens[start:end], w).float()
            logits = 15.0 * torch.tanh(logits / 15.0)
            losses = F.cross_entropy(
                logits, labels[start:end], ignore_index=-1, reduction="none"
            )
            if reduction == "none":
                pieces.append(losses)
            else:
                total = total + losses.sum()
        if reduction == "none":
            return torch.cat(pieces, dim=0)
        if reduction == "mean":
            # Do not clamp: PyTorch's mean on entirely ignored targets is NaN.
            total = total / (labels != -1).sum()
        return total


@_chunked_ce.register_fake
def _chunked_ce_fake(x, weight, targets, vocab: int, rows: int, reduction: str):
    shape = (targets.numel(),) if reduction == "none" else ()
    return x.new_empty(shape, dtype=torch.float32)


@torch.library.custom_op("nanochat::chunked_lm_ce_backward", mutates_args=())
def _chunked_ce_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    grad_loss: torch.Tensor,
    vocab: int,
    rows: int,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute only a logits tile and apply the exact softcap chain rule.

    The BF16 matmul adjoints are cast back to the master's dtype, as for the
    original F.linear(x, weight.to(x.dtype)). Accumulate dW in FP32.
    """
    tokens = x.reshape(-1, x.shape[-1])
    labels = targets.reshape(-1)
    w = weight[:vocab].to(tokens.dtype)
    dx = torch.empty_like(tokens)
    dw = torch.zeros_like(weight)
    if reduction == "mean":
        scale = grad_loss / (labels != -1).sum()
    elif reduction == "sum":
        scale = grad_loss
    with torch.no_grad():
        for start in range(0, tokens.shape[0], rows):
            end = min(start + rows, tokens.shape[0])
            block = tokens[start:end]
            target = labels[start:end]
            raw = F.linear(block, w).float()
            squashed = torch.tanh(raw / 15.0)
            probs = torch.softmax(squashed * 15.0, dim=-1)
            live = target != -1
            indices = torch.arange(end - start, device=x.device)
            # Ignore-index rows are set to zero after subtraction.
            probs[indices, target.clamp_min(0)] -= live.to(probs.dtype)
            probs = probs * live.unsqueeze(-1)
            dlogits = probs * (1.0 - squashed.square())
            if reduction == "none":
                dlogits = dlogits * grad_loss[start:end, None]
            else:
                dlogits = dlogits * scale
            dlogits = dlogits.to(block.dtype)
            dx[start:end] = dlogits @ w
            dw[:vocab].add_((dlogits.transpose(0, 1) @ block).to(dw.dtype))
    return dx.reshape_as(x), dw


@_chunked_ce_backward.register_fake
def _chunked_ce_backward_fake(x, weight, targets, grad_loss, vocab, rows, reduction):
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_context(ctx, inputs, output):
    x, weight, targets, vocab, rows, reduction = inputs
    ctx.save_for_backward(x, weight, targets)
    ctx.vocab, ctx.rows, ctx.reduction = vocab, rows, reduction


def _backward(ctx, grad_loss):
    x, weight, targets = ctx.saved_tensors
    dx, dw = _chunked_ce_backward(
        x, weight, targets, grad_loss.contiguous(),
        ctx.vocab, ctx.rows, ctx.reduction,
    )
    return dx, dw, None, None, None, None


_chunked_ce.register_autograd(_backward, setup_context=_setup_context)


def chunked_lm_cross_entropy(x, weight, targets, vocab, rows=2048, reduction="mean"):
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("unsupported cross-entropy reduction")
    if rows < 128 or rows > 8192:
        raise ValueError("NANOCHAT_CHUNKED_CE_ROWS must be between 128 and 8192")
    return _chunked_ce(x.contiguous(), weight, targets.contiguous(), vocab, rows, reduction)
