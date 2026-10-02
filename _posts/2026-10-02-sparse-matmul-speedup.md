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

We tested this on an H100 and a B200. The hypothesis holds, and so does its explanation: **if the SM clock is locked to a value the GPU can sustain under its power limit, the speedup disappears completely. Lock it higher than that and it comes back.**

_The code and raw results are available [here](https://github.com/GeometricAGI/blog/tree/main/sparse-matmul-speedup)._

### Setup

- `x @ w` in bf16 with `w` of shape 8192x8192 and `x` of shape `M x 8192`, run with plain `torch.matmul` (cuBLAS). No sparse kernels are involved; the zeros are stored densely.
- We set exactly that fraction of `w` to zero (5%, 10%, 25%, 50%, 70%, 90%, 99%), at positions chosen uniformly at random; the measured zero fraction matches to within 2e-7 and is stored in the results. Magnitude pruning (zeroing the smallest-magnitude weights) is also in the repo; it agrees at 99% zeros, but at 5% it is consistently faster than random zeroing on both GPUs (about 1.05-1.10x), which we have not explained, so we only tabulate random zeroing here. Everything below is "speedup over the dense weight", i.e. dense time divided by time with zeros, so values above 1.000 mean faster than dense.
- Matmuls are captured in a CUDA graph and timed over replays. Every config is timed in 5 interleaved rounds (alternating order) and we report the median, so drift hits every config equally.
- `M` is swept over 1024, 2048, 4096, 8192 and 16384. In an exploratory run on the H100 with smaller `M` (1, 16 and 128 rows) the matmul is memory-bound and sparsity made no difference (within 0.5%), so we start the published sweep at 1024.
- Each GPU is run with free-running clocks and with the SM clock locked (`nvidia-smi -lgc`) at six values: 1000, 1200, 1400, 1600, 1800 MHz and the GPU's maximum (1980 MHz on the H100, 1965 MHz on the B200).

### Free-running clocks: dense matmuls do get faster

Speedup with 99% of the weights zeroed, free-running clocks:

| | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| H100 | 1.206 | 1.149 | 1.112 | 1.092 | 1.111 |
| B200 | 1.075 | 1.068 | 1.074 | 1.065 | 1.082 |

![H100, free-running clocks](/assets/sparse-matmul-speedup/h100-unlocked.png)

![B200, free-running clocks](/assets/sparse-matmul-speedup/b200-unlocked.png)

- Speedup generally grows with the amount of zeros, up to about **1.21x on the H100 and 1.08x on the B200**.
- At realistic sparsity (5-25%) the effect is mostly inside the run-to-run noise. On the H100 at `M=16384`, 5-50% zeros was 3-7% **slower** than dense.
- Differences of a couple of percent between neighbouring points are within run-to-run noise.

### Locked clocks: the speedup depends on the lock

Speedup with 99% zeros for each locked SM clock (the first row is the free-running result for reference):

**H100**

| SM clock | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| free-running | 1.206 | 1.149 | 1.112 | 1.092 | 1.111 |
| 1000 MHz | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 |
| 1200 MHz | 1.000 | 1.001 | 1.000 | 1.000 | 1.000 |
| 1400 MHz | 1.000 | 1.001 | 1.024 | 1.001 | 1.050 |
| 1600 MHz | 1.175 | 1.152 | 1.136 | 1.120 | 1.162 |
| 1800 MHz | 1.196 | 1.147 | 1.122 | 1.113 | 1.162 |
| 1980 MHz | 1.189 | 1.155 | 1.139 | 1.101 | 1.160 |

**B200**

| SM clock | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| free-running | 1.075 | 1.068 | 1.074 | 1.065 | 1.082 |
| 1000 MHz | 1.012 | 1.008 | 1.006 | 1.001 | 0.999 |
| 1200 MHz | 1.002 | 1.003 | 1.000 | 1.001 | 1.000 |
| 1400 MHz | 1.044 | 1.063 | 1.051 | 1.070 | 1.099 |
| 1600 MHz | 1.063 | 1.085 | 1.097 | 1.068 | 1.075 |
| 1800 MHz | 1.099 | 1.069 | 1.072 | 1.090 | 1.053 |
| 1965 MHz | 1.109 | 1.076 | 1.083 | 1.100 | 1.072 |

![H100, speedup vs. locked clock](/assets/sparse-matmul-speedup/h100-clock_sweep.png)

![B200, speedup vs. locked clock](/assets/sparse-matmul-speedup/b200-clock_sweep.png)

- **Low locks: no effect.** At 1000 and 1200 MHz the H100 is within 0.1% of dense for every `M` and every sparsity level. The B200 is within 1.2% at 1000 MHz and 0.3% at 1200 MHz. The GPU holds the requested clock, so the work takes the same time whatever is in the weights.
- **High locks: the speedup comes back.** On the H100 it is partly back at 1400 MHz (1.00-1.05x) and fully back, at its free-running size, from 1600 MHz up. On the B200 it is back from 1400 MHz. Above that the lock stops being the binding constraint: dense runs reach only 1260-1410 MHz on the H100 and 1155-1402 MHz on the B200 no matter how high we set the lock, because the power limit decides the clock. Sparser weights draw less power, so they sustain a somewhat higher clock.
- A clock lock is therefore only a control if it is set below the clock the GPU would sustain at its power limit for that workload.

### Full results

The tables below give the speedup for every `M` and sparsity level for each clock setting.

<details><summary>H100, free-running</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.033 | 1.033 | 1.058 | 1.109 | 1.158 | 1.187 | 1.206 |
| 2048 | 0.992 | 0.992 | 1.001 | 1.027 | 1.056 | 1.115 | 1.149 |
| 4096 | 0.990 | 0.987 | 0.992 | 0.992 | 1.017 | 1.099 | 1.112 |
| 8192 | 1.007 | 1.000 | 1.018 | 1.022 | 1.029 | 1.086 | 1.092 |
| 16384 | 0.942 | 0.926 | 0.928 | 0.969 | 1.001 | 1.083 | 1.111 |

</details>

<details><summary>H100, 1000 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16384 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 1.001 | 1.001 |

</details>

<details><summary>H100, 1200 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 |
| 2048 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16384 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 |

</details>

<details><summary>H100, 1400 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 |
| 2048 | 0.998 | 0.998 | 0.999 | 1.000 | 1.001 | 1.001 | 1.001 |
| 4096 | 1.024 | 1.024 | 1.024 | 1.024 | 1.024 | 1.024 | 1.024 |
| 8192 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 |
| 16384 | 1.016 | 1.027 | 1.035 | 1.004 | 1.044 | 1.051 | 1.050 |

</details>

<details><summary>H100, 1600 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.012 | 1.012 | 1.024 | 1.061 | 1.130 | 1.158 | 1.175 |
| 2048 | 0.998 | 0.985 | 0.991 | 1.018 | 1.039 | 1.129 | 1.152 |
| 4096 | 0.990 | 0.984 | 0.993 | 1.011 | 1.038 | 1.130 | 1.136 |
| 8192 | 0.997 | 0.989 | 1.002 | 1.013 | 1.055 | 1.123 | 1.120 |
| 16384 | 0.969 | 0.956 | 0.956 | 0.991 | 1.028 | 1.129 | 1.162 |

</details>

<details><summary>H100, 1800 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.007 | 1.032 | 1.054 | 1.087 | 1.120 | 1.175 | 1.196 |
| 2048 | 0.984 | 0.984 | 0.990 | 1.015 | 1.033 | 1.128 | 1.147 |
| 4096 | 1.002 | 1.002 | 1.001 | 1.007 | 1.034 | 1.119 | 1.122 |
| 8192 | 0.999 | 0.998 | 1.005 | 1.041 | 1.044 | 1.105 | 1.113 |
| 16384 | 0.971 | 0.951 | 0.956 | 0.973 | 1.027 | 1.124 | 1.162 |

</details>

<details><summary>H100, 1980 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.009 | 1.021 | 1.040 | 1.085 | 1.115 | 1.163 | 1.189 |
| 2048 | 0.993 | 0.994 | 1.002 | 1.028 | 1.045 | 1.134 | 1.155 |
| 4096 | 1.000 | 0.991 | 0.996 | 1.006 | 1.033 | 1.115 | 1.139 |
| 8192 | 1.014 | 1.012 | 1.020 | 1.026 | 1.061 | 1.134 | 1.101 |
| 16384 | 0.967 | 0.947 | 0.957 | 0.977 | 1.030 | 1.119 | 1.160 |

</details>

<details><summary>B200, free-running</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.001 | 1.023 | 1.039 | 1.064 | 1.064 | 1.075 |
| 2048 | 0.992 | 0.991 | 1.009 | 1.027 | 1.037 | 1.048 | 1.068 |
| 4096 | 1.001 | 1.006 | 1.022 | 1.051 | 1.072 | 1.073 | 1.074 |
| 8192 | 0.994 | 0.995 | 1.012 | 1.019 | 1.056 | 1.043 | 1.065 |
| 16384 | 0.996 | 1.001 | 1.006 | 1.026 | 1.049 | 1.069 | 1.082 |

</details>

<details><summary>B200, 1000 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.001 | 1.001 | 1.004 | 1.007 | 1.011 | 1.012 |
| 2048 | 1.000 | 1.000 | 1.001 | 1.002 | 1.005 | 1.008 | 1.008 |
| 4096 | 1.000 | 1.000 | 1.001 | 1.001 | 1.003 | 1.005 | 1.006 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.001 | 1.002 | 1.001 | 1.001 |
| 16384 | 1.000 | 1.000 | 1.000 | 1.001 | 1.001 | 0.999 | 0.999 |

</details>

<details><summary>B200, 1200 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.001 | 0.999 | 0.999 | 1.000 | 1.000 | 1.002 | 1.002 |
| 2048 | 1.000 | 1.000 | 1.001 | 1.001 | 1.002 | 1.003 | 1.003 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.001 |
| 16384 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

</details>

<details><summary>B200, 1400 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.020 | 1.041 | 1.042 | 1.042 | 1.043 | 1.044 | 1.044 |
| 2048 | 1.017 | 1.013 | 1.019 | 1.025 | 1.043 | 1.059 | 1.063 |
| 4096 | 0.999 | 1.000 | 1.012 | 1.019 | 1.044 | 1.068 | 1.051 |
| 8192 | 1.022 | 1.073 | 1.029 | 1.041 | 1.068 | 1.067 | 1.070 |
| 16384 | 1.000 | 1.005 | 1.014 | 1.022 | 1.033 | 1.054 | 1.099 |

</details>

<details><summary>B200, 1600 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.007 | 1.014 | 1.024 | 1.034 | 1.048 | 1.058 | 1.063 |
| 2048 | 1.004 | 1.012 | 1.019 | 1.023 | 1.037 | 1.052 | 1.085 |
| 4096 | 1.005 | 1.009 | 1.024 | 1.034 | 1.073 | 1.068 | 1.097 |
| 8192 | 1.002 | 1.020 | 1.058 | 1.024 | 1.055 | 1.046 | 1.068 |
| 16384 | 0.989 | 1.001 | 1.014 | 1.021 | 1.028 | 1.050 | 1.075 |

</details>

<details><summary>B200, 1800 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.002 | 1.014 | 1.017 | 1.040 | 1.049 | 1.068 | 1.099 |
| 2048 | 1.003 | 1.004 | 1.008 | 1.011 | 1.041 | 1.068 | 1.069 |
| 4096 | 1.000 | 1.001 | 1.015 | 1.025 | 1.050 | 1.078 | 1.072 |
| 8192 | 1.022 | 1.018 | 1.023 | 1.038 | 1.047 | 1.064 | 1.090 |
| 16384 | 0.988 | 0.979 | 1.010 | 1.020 | 1.057 | 1.058 | 1.053 |

</details>

<details><summary>B200, 1965 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.019 | 1.040 | 1.066 | 1.049 | 1.109 |
| 2048 | 1.002 | 1.004 | 1.013 | 1.015 | 1.029 | 1.052 | 1.076 |
| 4096 | 1.004 | 1.012 | 1.013 | 1.021 | 1.051 | 1.035 | 1.083 |
| 8192 | 1.011 | 1.011 | 1.025 | 1.026 | 1.050 | 1.083 | 1.100 |
| 16384 | 1.004 | 1.005 | 1.009 | 1.025 | 1.040 | 1.055 | 1.072 |

</details>

Magnitude-pruning results and raw timings, SM clocks and power samples are in [the repo](https://github.com/GeometricAGI/blog/tree/main/sparse-matmul-speedup/results).

### What this means

- **It is a power effect, not a compute effect.** The kernel does the same work. It only runs faster when the GPU is power-limited and the zeros let it hold a higher clock; at a clock the GPU can sustain, the time is identical.
- **Benchmark with care.** If you compare kernels or models and one of them runs on weights with many zeros (pruned, quantized, or just initialized differently), part of the gap can be the power cap rather than the kernel. Lock clocks to a value below the power-limited clock, or at least log SM clock and power next to your timings.
- **Don't count on it.** The effect is small, only shows up for compute-bound shapes at power-limited clocks, and is smaller than the noise at the sparsity levels that are common in practice.

Limitations: one weight shape (8192x8192), bf16 only, one GPU of each type, and power readings from NVML are sampled, not integrated. A natural follow-up is to compare against a genuine sparse kernel path such as cuSPARSELt, and to repeat this for fp8.
