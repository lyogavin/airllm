"""Fused linear-cross-entropy that never materialises ``[N, vocab]`` logits.

Qwen3.8's lm_head is 248320 × 5120 (2.54GB bf16). A naive ``hidden @ W.T`` at
seq 2048 is another ~2GB of fp32 logits. This walks the vocabulary in chunks,
keeps a running log-sum-exp, and recomputes the same chunks in backward to get
``d_hidden``. ``W`` may live on CPU.
"""

import torch


def _compute_device(hidden):
    """Matmuls follow ``hidden`` unless it is on CPU and CUDA is available."""
    if hidden.is_cuda or not torch.cuda.is_available():
        return hidden.device
    return torch.device("cuda")


class _ChunkedCE(torch.autograd.Function):
    """Tensor-only front door. ``weight`` is stashed on the class for one apply()."""

    weight = None
    chunk_size = 4096
    ignore_index = -100
    reduction = "mean"

    @staticmethod
    def forward(ctx, hidden, labels):
        weight = _ChunkedCE.weight
        chunk_size = int(_ChunkedCE.chunk_size)
        ignore_index = int(_ChunkedCE.ignore_index)
        reduction = _ChunkedCE.reduction
        if weight is None:
            raise RuntimeError("chunked CE weight was not set")

        compute_device = _compute_device(hidden)
        h = hidden.to(compute_device)
        n, _hidden = h.shape
        vocab = weight.shape[0]
        if chunk_size < 1:
            chunk_size = vocab

        labels = labels.reshape(-1).to(device=compute_device)
        if labels.numel() != n:
            raise ValueError(f"labels has {labels.numel()} elements, hidden has {n} rows")

        valid = labels != ignore_index
        logz = torch.full((n,), float("-inf"), device=compute_device, dtype=torch.float32)
        target_logit = torch.zeros(n, device=compute_device, dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, vocab, chunk_size):
                end = min(start + chunk_size, vocab)
                w = weight[start:end].to(device=compute_device, dtype=h.dtype)
                logits = (h @ w.t()).float()
                logz = torch.logaddexp(logz, logits.logsumexp(dim=-1))
                in_chunk = valid & (labels >= start) & (labels < end)
                if in_chunk.any():
                    rows = in_chunk.nonzero(as_tuple=False).squeeze(-1)
                    target_logit[rows] = logits[rows, labels[rows] - start]

        n_valid = valid.sum().clamp(min=1).to(dtype=torch.float32)
        per_token = torch.where(valid, logz - target_logit, torch.zeros_like(logz))
        if reduction == "sum":
            loss = per_token.sum()
        elif reduction == "none":
            loss = per_token
        else:
            loss = per_token.sum() / n_valid

        ctx.save_for_backward(hidden, labels.to(hidden.device))
        ctx.weight = weight
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.n_valid = int(n_valid.item())
        ctx.valid = valid.to(hidden.device)
        ctx.compute_device = compute_device
        ctx.reduction = reduction
        ctx.logz = logz.to(hidden.device)
        return loss if hidden.is_cuda or loss.device == hidden.device else loss.to(hidden.device)

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, labels = ctx.saved_tensors
        weight = ctx.weight
        compute_device = ctx.compute_device
        h = hidden.to(compute_device)
        labels = labels.to(compute_device)
        valid = ctx.valid.to(compute_device)
        logz = ctx.logz.to(compute_device)
        vocab = weight.shape[0]
        chunk_size = ctx.chunk_size

        scale = grad_loss.to(compute_device)
        if ctx.reduction == "mean":
            scale = scale / max(ctx.n_valid, 1)
        elif ctx.reduction == "none":
            scale = scale.reshape(-1)

        d_hidden = torch.zeros(h.shape, device=compute_device, dtype=torch.float32)

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w = weight[start:end].to(device=compute_device, dtype=h.dtype)
            logits = (h @ w.t()).float()
            probs = torch.exp(logits - logz.unsqueeze(-1))
            in_chunk = valid & (labels >= start) & (labels < end)
            if in_chunk.any():
                rows = in_chunk.nonzero(as_tuple=False).squeeze(-1)
                probs[rows, labels[rows] - start] -= 1.0
            probs = probs.masked_fill(~valid.unsqueeze(-1), 0.0)
            if ctx.reduction == "none":
                d_hidden += (probs * scale.unsqueeze(-1)) @ w.float()
            else:
                d_hidden += (probs @ w.float()) * scale

        d_hidden = d_hidden.to(dtype=hidden.dtype, device=hidden.device)
        return d_hidden, None


def chunked_linear_cross_entropy(
    hidden,
    weight,
    labels,
    chunk_size=4096,
    ignore_index=-100,
    reduction="mean",
):
    """Cross-entropy of ``hidden @ weight.T`` vs ``labels``, chunked over vocab.

    ``hidden`` is ``[N, H]`` or ``[B, S, H]`` and may live on CPU. ``weight`` is
    ``[V, H]`` and may live on CPU. ``labels`` matches hidden's token layout.
    """
    if hidden.dim() == 3:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    _ChunkedCE.weight = weight
    _ChunkedCE.chunk_size = chunk_size
    _ChunkedCE.ignore_index = ignore_index
    _ChunkedCE.reduction = reduction
    try:
        return _ChunkedCE.apply(hidden, labels)
    finally:
        _ChunkedCE.weight = None
