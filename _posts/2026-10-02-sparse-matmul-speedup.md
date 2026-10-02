---
layout: post
title: "Zeros make dense matmuls faster, but not the way you'd hope"
description: "Zeroing out weights speeds up a plain dense cuBLAS matmul by up to 14% on H100 and B200. Lock the clock and the speedup vanishes: the GPU is just spending less power."
date: 2026-10-02
author:
  name: Pramodith B
  title: Member of Technical Staff
  linkedin: https://www.linkedin.com/in/pramodith/
---

*By [Pramodith B](https://www.linkedin.com/in/pramodith/), Member of Technical Staff*

## Zeros make dense matmuls faster, but not the way you'd hope

Hypothesis: if one operand of a matmul is sparse, even a **dense** kernel that knows nothing about sparsity should run faster. Nothing is skipped, but zeros flip fewer bits, so the chip draws less power. GPUs like the H100 and B200 run matmuls at their power cap, so lower power can mean higher clocks, which means faster matmuls.

We tested this on an H100 and a B200. The hypothesis holds, and so does its explanation: **with the SM clock locked, the speedup disappears completely.**

_The code and raw results are available [here](https://github.com/GeometricAGI/blog/tree/main/sparse-matmul-speedup)._

### Setup

- `x @ w` in bf16 with `w` of shape 8192x8192 and `x` of shape `M x 8192`, run with plain `torch.matmul` (cuBLAS). No sparse kernels are involved; the zeros are stored densely.
- We zero a fraction of `w` (5%, 10%, 25%, 50%, 70%, 90%, 99%), either at random or by magnitude, and also test a 2:4 pattern (zero the 2 smallest of every 4 consecutive entries along the contraction dimension). 2:4 is a fixed pattern, so it is always exactly 50% sparse; it is the only pattern NVIDIA's sparse tensor cores accelerate.
- Matmuls are captured in a CUDA graph and timed over replays. Every config is timed in 5 interleaved rounds (alternating order) and we report the median, so drift hits every config equally.
- `M` is swept from 1024 to 16384. In an exploratory run on the H100 with smaller `M` (1, 16 and 128 rows) the matmul is memory-bound and sparsity made no difference (within 0.5%), so we start the published sweep at 1024.
- We run each GPU twice: with free-running clocks, and with the SM clock locked (`nvidia-smi -lgc`) to a value the GPU can hold without hitting its power cap (1200 MHz on the H100, 1000 MHz on the B200).

### Free-running clocks: dense matmuls do get faster

Speedup over the dense weight, random zeroing, `M=4096`:

| | 5% | 10% | 25% | 50% | 70% | 90% | 99% | 2:4 |
|---|---|---|---|---|---|---|---|---|
| H100 | 1.014 | 1.032 | 1.015 | 1.051 | 1.069 | 1.098 | 1.145 | 1.095 |
| B200 | 0.990 | 0.997 | 1.004 | 1.013 | 1.115 | 1.062 | 1.054 | 1.113 |

![H100, free-running clocks](/assets/sparse-matmul-speedup/h100-unlocked.png)

![B200, free-running clocks](/assets/sparse-matmul-speedup/b200-unlocked.png)

- Speedup generally grows with the amount of zeros, up to about **1.14x on the H100 and 1.13x on the B200** at extreme sparsity.
- The 2:4 pattern at 50% zeros reaches up to 1.13x (B200, `M=16384`) and 1.10x (H100, `M=4096`), but only 1.00x on the H100 at `M=16384`. It is still the same dense kernel; the speedup is not the sparse tensor-core path.
- At realistic sparsity (5-25%) the effect is mostly inside the run-to-run noise. At `M=16384` on the H100, 5-50% sparsity was actually 3-4% **slower** than dense.
- The curves are noisy. Magnitude-pruning at 5% was faster than at 10% on both GPUs, which we can't explain with the clock data alone, so treat differences of a few percent between neighbouring points as noise.

### Locked clocks: the speedup is gone

Same sweep with the SM clock pinned (H100 at 1200 MHz, B200 at 1000 MHz), speedup over dense for random zeroing:

| | M | 5% | 25% | 50% | 99% | 2:4 |
|---|---|---|---|---|---|---|
| H100 | 1024 | 1.000 | 1.000 | 1.001 | 1.001 | 1.000 |
| H100 | 16384 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 |
| B200 | 1024 | 1.000 | 1.001 | 1.002 | 1.003 | 1.001 |
| B200 | 16384 | 1.000 | 1.000 | 1.001 | 1.001 | 1.001 |

![H100, locked clock](/assets/sparse-matmul-speedup/h100-locked-1200mhz.png)

![B200, locked clock](/assets/sparse-matmul-speedup/b200-locked-1000mhz.png)

Across every `M` and every sparsity level (including 99% and 2:4), the largest difference to dense is **0.4%**. Even the odd points from the unlocked runs flatten out, so they were clock-driven noise too.

### Full results

Speedup over the dense weight (dense time / time with zeros) for random zeroing, for every `M`. Values above 1.000 are faster than dense. The 2:4 column is a fixed pattern (2 zeros in every group of 4 consecutive entries), so it is always exactly 50% sparse and there is no other sparsity level to report for it.

**H100, free-running clocks**

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% | 2:4 (50%) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1.016 | 1.013 | 1.047 | 1.055 | 1.097 | 1.132 | 1.120 | 1.068 |
| 2048 | 1.014 | 1.012 | 1.039 | 1.041 | 1.065 | 1.092 | 1.098 | 1.092 |
| 4096 | 1.014 | 1.032 | 1.015 | 1.051 | 1.069 | 1.098 | 1.145 | 1.095 |
| 8192 | 1.013 | 1.017 | 1.034 | 1.050 | 1.084 | 1.098 | 1.127 | 1.100 |
| 16384 | 0.970 | 0.957 | 0.970 | 0.963 | 1.017 | 1.093 | 1.140 | 1.003 |

**H100, SM clock locked at 1200 MHz**

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% | 2:4 (50%) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 0.998 | 1.000 | 1.001 | 1.000 | 1.001 | 1.001 | 1.000 |
| 2048 | 1.001 | 0.999 | 1.002 | 1.000 | 1.003 | 1.004 | 1.004 | 1.003 |
| 4096 | 0.998 | 1.001 | 1.001 | 1.000 | 1.001 | 1.001 | 1.002 | 1.001 |
| 8192 | 1.001 | 1.001 | 1.001 | 1.002 | 1.001 | 1.002 | 1.002 | 1.002 |
| 16384 | 1.000 | 1.001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 |

**B200, free-running clocks**

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% | 2:4 (50%) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.003 | 1.019 | 1.024 | 1.031 | 1.054 | 1.077 | 1.078 |
| 2048 | 1.000 | 1.000 | 1.012 | 1.020 | 1.026 | 1.043 | 1.051 | 1.084 |
| 4096 | 0.990 | 0.997 | 1.004 | 1.013 | 1.115 | 1.062 | 1.054 | 1.113 |
| 8192 | 0.999 | 1.010 | 1.017 | 1.038 | 1.067 | 1.080 | 1.081 | 1.114 |
| 16384 | 1.000 | 1.014 | 1.030 | 1.066 | 1.073 | 1.104 | 1.099 | 1.125 |

**B200, SM clock locked at 1000 MHz**

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% | 2:4 (50%) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.001 | 1.001 | 1.002 | 1.003 | 1.003 | 1.003 | 1.001 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.001 | 1.001 | 1.001 | 1.002 | 1.001 |
| 4096 | 1.000 | 1.000 | 1.001 | 1.002 | 1.002 | 1.001 | 1.001 | 1.001 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.001 | 1.001 | 1.001 | 1.000 | 1.001 |
| 16384 | 1.000 | 1.000 | 1.000 | 1.001 | 1.002 | 1.001 | 1.001 | 1.001 |

Magnitude-pruning results and raw timings, SM clocks and power samples are in [the repo](https://github.com/GeometricAGI/blog/tree/main/sparse-matmul-speedup/results).

### What this means

- **It is a power effect, not a compute effect.** The kernel does the same work and takes the same time at a fixed clock. Unlocked, the dense runs sat at the power cap (about 700 W on the H100 and 990 W on the B200) with SM clocks dropping to 1150-1400 MHz; sparse weights let the GPU hold a somewhat higher clock.
- **Benchmark with care.** If you compare kernels or models and one of them runs on weights with many zeros (pruned, quantized, or just initialized differently), part of the gap can be the power cap rather than the kernel. Lock clocks, or at least log SM clock and power next to your timings.
- **Don't count on it.** The effect is small, only shows up for compute-bound shapes at power-limited clocks, and is smaller than the noise at the sparsity levels that are common in practice.

Limitations: one weight shape (8192x8192), bf16 only, one GPU of each type, and power readings from NVML are sampled, not integrated. A natural follow-up is to compare against a genuine sparse kernel path such as cuSPARSELt for the 2:4 case, and to repeat this for fp8.
