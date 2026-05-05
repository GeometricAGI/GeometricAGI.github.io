---
layout: post
title: "Open-sourcing kernels on the Hugging Face Hub"
description: "The Geometric team is open-sourcing 6 loss function kernels on the Hugging Face Hub."
date: 2026-05-07
author:
  - name: "Fionnán Alt"
    title: "Member of Technical Staff"
    linkedin: "https://www.linkedin.com/in/fionnanalt/"
  - name: "Jack Foxabbot"
    title: "Member of Technical Staff"
    linkedin: "https://www.linkedin.com/in/foxabbott/"
  - name: "Pramodith B"
    title: "Member of Technical Staff"
    linkedin: "https://www.linkedin.com/in/pramodith/"
---

# Open-sourcing kernels on the Hugging Face Hub

In the past few months at Geometric, we've been learning what it takes to create high-performance kernels. Today we're releasing six of these kernels on the Hugging Face Hub with a commitment to open-source more in the future. The kernel card is available [here](https://huggingface.co/Geometric-AI/geometric-ai-kernels) and can be used via Hugging Face's `kernels` library.

### Kernels

The six kernels we are releasing include:

- GRPO, BNPO, Reverse KL Forward: compute the GRPO, BNPO, and Reverse KL Forward loss functions. These kernels compute only the forward pass and can be used in validation/testing.
- GRPO, BNPO, Reverse KL Fused Forward+Backward: compute GRPO, BNPO, and Reverse KL with a fused forward and backward pass. These kernels return both the loss and the gradient of the model being optimized and can be used in training.

All kernels are implemented in **CuteDSL** and support `fp32`, `fp16`, and `bf16`. They have been profiled on H100 GPUs and should run on Ampere and Blackwell architectures as well.

Our kernels support dynamic sequence lengths so we avoid re-compiling the kernel each time the sequence length changes, which is common in post-training workflows.

### Using our kernels

Each kernel family has three variants:

1. A pure forward kernel with suffix `_fwd`.
2. A fused forward+backward kernel with suffix `_loss` that returns a tuple `(loss, grad)`.
3. An autograd-aware wrapper with suffix `_autograd` that returns just the loss but supports `loss.backward()`. __The autograd-aware wrapper has a noticeable over head compared to 2__, we'll continue to investigate ways to reduce this overhead in the future.

To use our kernels, install the `kernels` library and import a kernel module:

```python
# make sure `kernels` is installed: pip install -U kernels
from kernels import get_kernel

kernel_module = get_kernel("Geometric-AI/geometric-ai-kernels")
grpo_loss = kernel_module.grpo_loss_fwd

grpo_loss(...)
```

#### Training with our kernels

There are two ways to use our kernels for training:

1) Direct `(loss, grad)` return - the lowest-overhead path.

The fused forward+backward kernel writes both the scalar loss and the closed-form gradient `dL/d(policy_logprobs)` in a single CUDA launch. The Python wrapper returns them as a tuple, and the caller chains the gradient into the upstream model:

```python
loss, grad_policy = kernel_module.grpo_loss(
    policy_logprobs,
    old_policy_logprobs,
    ref_logprobs,
    advantages,
    completions_mask,
)
policy_logprobs.backward(grad_policy)
```
Note that the returned `loss` is a detached scalar tensor, so you will need to use the returned `grad_policy` to backpropagate through the rest of the model. This is the lowest-overhead way to use our kernels for training.

2) Autograd-aware `_autograd` - drop-in for an eager training loop.

If you want `loss.backward()` since it's more compatible with most training libraries use the `_autograd` variants. They wrap the same fused kernel as a `torch.library` custom op with a registered backward:

```python
loss = kernel_module.grpo_loss_autograd(
    policy_logprobs,
    old_policy_logprobs,
    ref_logprobs,
    advantages,
    completions_mask,
)
loss.backward()
```

## Benchmarking our kernels
In order to ensure a fair comparision of our cute kernels vs torch.compile kernels we investigated two ways of creating fused forward+backward kernels in torch.

### Method 1: Using `torch.compile` with `fullgraph=True` and `torch.dynamo.config.trace_autograd_ops`
We set the fullgraph flag in `torch.compile` to True and enabled `torch.dynamo.config.trace_autograd_ops` to allow autograd operations to be traced and fused together. This method allows for the creation of fused forward+backward kernels. A fused kernel can then be created as follows:

```python

def torch_reverse_kl_div(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completions_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plain-Python reverse-KL reference traceable by AOTAutograd / Inductor.

    Mirrors ``geo_evo.torch_kernels.torch_reverse_kl_div_baseline``.
    Computes ``KL(student || teacher) = sum_v p(v) [log p(v) - log q(v)]``
    averaged over valid tokens.
    """
    log_p = log_softmax(student_logits, dim=-1)
    log_q = log_softmax(teacher_logits, dim=-1)

    # ``kl_div(input, target, log_target=True)`` computes
    # ``KL(target || input)``, so input=log_q, target=log_p gives
    # ``KL(student || teacher)``.
    kl = kl_div(log_q, log_p, log_target=True, reduction="none").sum(dim=-1)

    if completions_mask is not None:
        # Cast n_valid to fp32: int64 → fp16 overflows when
        # n_valid > 65504. ``clamp(min=1.0)`` matches the cute kernel's
        # ``cute.arch.fmax(..., 1.0)`` before ``rcp_approx`` — a
        # fully-masked batch produces ``loss=0`` instead of inf/NaN.
        n_valid = completions_mask.sum().to(torch.float32).clamp(min=1.0)
        kl = (kl * completions_mask).sum() / n_valid
    else:
        kl = kl.mean()

    return kl.to(student_logits.dtype)

def torch_reverse_kl_div_grad(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completions_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the gradient and loss so"""
    loss = torch_reverse_kl_div(student_logits, teacher_logits, completions_mask=completions_mask)
    (grad_student,) = torch.autograd.grad(loss, [student_logits])
    return loss.detach(), grad_student

fused_reverse_kl = torch.compile(torch_reverse_kl_div_grad, fullgraph=True, mode="max-autotune-no-cudagraphs")
```

## Method 2: Writing out the forward and backward ops in a single function
We can also write out the forward and backward pass in a single, for reverse KL this would look like:

```python
def torch_reverse_kl_div_fwd_bwd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completions_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a reverse-KL forward + backward step using only forward ops.

    Args:
        student_logits: Student logits, shape (N, C, V). ``requires_grad``
            is ignored — gradients are computed manually.
        teacher_logits: Teacher logits, shape (N, C, V).
        completions_mask: Boolean mask of shape (N, C). Pass an all-ones
            mask if no tokens are padded.

    Returns:
        Tuple ``(loss, grad_student_logits)``.
    """
    with torch.no_grad():
        log_p = log_softmax(student_logits, dim=-1)
        log_q = log_softmax(teacher_logits, dim=-1)
        p = torch.exp(log_p)

        log_diff = log_p - log_q  # (N, C, V)
        kl_per_tok = (p * log_diff).sum(dim=-1)  # (N, C)

        # Cast n_valid to fp32 to avoid fp16 overflow when n_valid > 65504.
        n_valid = completions_mask.sum().to(torch.float32)
        inv_n_fp32 = torch.reciprocal(n_valid)
        mask = completions_mask.to(student_logits.dtype)
        # Loss aggregates in fp32 (sum_fp16 * inv_n_fp32 promotes to fp32);
        # grad uses the low-precision reciprocal so the per-tok elementwise
        # stays in the input dtype.
        loss = (kl_per_tok * mask).sum() * inv_n_fp32
        inv_n = inv_n_fp32.to(student_logits.dtype)

        # Per-token gradient: p_k * (log_p_k - log_q_k - kl). Broadcast kl
        # over the vocab dim. The +1 terms from differentiating sum_v p_v
        # cancel, so no constant offset appears here.
        grad_per_tok = p * (log_diff - kl_per_tok.unsqueeze(-1))  # (N, C, V)
        grad_student = grad_per_tok * mask.unsqueeze(-1) * inv_n

    return loss.to(student_logits.dtype), grad_student
```

In order to ensure that torch.compile correctly compiled both kernels in the same manner we profiled both variants and **found that they had very similar run-time profiles** showing that both methods successfully created fused forward+backward kernels.

## Benchmarking Results
