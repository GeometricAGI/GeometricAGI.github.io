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
- We set a fraction of `w` to zero (5%, 10%, 25%, 50%, 70%, 90%, 99%), chosen uniformly at random. Pruning by magnitude instead gives similar numbers and is in the repo. Everything below is "speedup over the dense weight", i.e. dense time divided by time with zeros, so values above 1.000 mean faster than dense.
- Matmuls are captured in a CUDA graph and timed over replays. Every config is timed in 5 interleaved rounds (alternating order) and we report the median, so drift hits every config equally.
- `M` is swept over 1024, 2048, 4096, 8192 and 16384. In an exploratory run on the H100 with smaller `M` (1, 16 and 128 rows) the matmul is memory-bound and sparsity made no difference (within 0.5%), so we start the published sweep at 1024.
- Each GPU is run with free-running clocks and with the SM clock locked (`nvidia-smi -lgc`) at six values: 1000, 1200, 1400, 1600, 1800 MHz and the GPU's maximum (1980 MHz on the H100, 1965 MHz on the B200).

### Free-running clocks: dense matmuls do get faster

Speedup with 99% of the weights zeroed, free-running clocks:

| | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| H100 | 1.184 | 1.113 | 1.104 | 1.077 | 1.129 |
| B200 | 1.088 | 1.078 | 1.100 | 1.080 | 1.049 |

![H100, free-running clocks](/assets/sparse-matmul-speedup/h100-unlocked.png)

![B200, free-running clocks](/assets/sparse-matmul-speedup/b200-unlocked.png)

- Speedup generally grows with the amount of zeros, up to about **1.18x on the H100 and 1.12x on the B200**.
- At realistic sparsity (5-25%) the effect is mostly inside the run-to-run noise. On the H100 at `M=16384`, 5-50% zeros was 3-6% **slower** than dense.
- The curves are noisy, so treat differences of a few percent between neighbouring points as noise.

### Locked clocks: the speedup depends on the lock

Speedup with 99% zeros for each locked SM clock (the first row is the free-running result for reference):

**H100**

| SM clock | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| free-running | 1.184 | 1.113 | 1.104 | 1.077 | 1.129 |
| 1000 MHz | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1200 MHz | 1.000 | 1.001 | 1.000 | 1.000 | 1.000 |
| 1400 MHz | 1.001 | 1.016 | 1.076 | 1.022 | 1.048 |
| 1600 MHz | 1.165 | 1.147 | 1.116 | 1.100 | 1.157 |
| 1800 MHz | 1.157 | 1.124 | 1.112 | 1.108 | 1.159 |
| 1980 MHz | 1.163 | 1.145 | 1.113 | 1.108 | 1.142 |

**B200**

| SM clock | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| free-running | 1.088 | 1.078 | 1.100 | 1.080 | 1.049 |
| 1000 MHz | 1.004 | 1.007 | 1.000 | 1.003 | 1.003 |
| 1200 MHz | 1.002 | 1.002 | 1.026 | 1.021 | 1.029 |
| 1400 MHz | 1.063 | 1.068 | 1.067 | 1.064 | 1.064 |
| 1600 MHz | 1.068 | 1.090 | 1.069 | 1.050 | 1.078 |
| 1800 MHz | 1.044 | 1.064 | 1.082 | 1.068 | 1.073 |
| 1965 MHz | 1.081 | 1.071 | 1.068 | 1.071 | 1.076 |

![H100, speedup vs. locked clock](/assets/sparse-matmul-speedup/h100-clock_sweep.png)

![B200, speedup vs. locked clock](/assets/sparse-matmul-speedup/b200-clock_sweep.png)

- **Low locks: no effect.** At 1000 and 1200 MHz the H100 is within 0.1% of dense for every `M` and every sparsity level. The B200 is within 0.7% at 1000 MHz. The GPU holds the requested clock, so the work takes the same time whatever is in the weights.
- **High locks: the speedup comes back.** On the H100 it is partly back at 1400 MHz (1.00-1.08x) and fully back, at its free-running size, from 1600 MHz up. On the B200 it is back from 1400 MHz. Above that the lock stops being the binding constraint: dense runs reach only 1260-1400 MHz on the H100 and 1155-1400 MHz on the B200 no matter how high we set the lock, because the power limit decides the clock. Sparser weights draw less power, so they sustain a somewhat higher clock.
- **B200 at 1200 MHz is in between.** Speedup is 1.00-1.03x for the larger shapes, so this lock sits right at the edge of what it can sustain.
- A clock lock is therefore only a control if it is set below the clock the GPU would sustain at its power limit for that workload.

### Full results

The tables below give the speedup for every `M` and sparsity level for each clock setting.

<details><summary>H100, free-running</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.028 | 1.032 | 1.040 | 1.083 | 1.126 | 1.171 | 1.184 |
| 2048 | 0.991 | 0.988 | 0.993 | 1.001 | 1.032 | 1.102 | 1.113 |
| 4096 | 0.998 | 0.987 | 0.962 | 0.992 | 1.013 | 1.066 | 1.104 |
| 8192 | 0.995 | 0.986 | 1.005 | 1.011 | 1.045 | 1.091 | 1.077 |
| 16384 | 0.965 | 0.944 | 0.944 | 0.967 | 1.021 | 1.089 | 1.129 |

</details>

<details><summary>H100, 1000 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16384 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 |

</details>

<details><summary>H100, 1200 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16384 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

</details>

<details><summary>H100, 1400 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.001 | 0.997 | 1.000 | 1.001 | 1.001 | 1.001 |
| 2048 | 1.016 | 1.012 | 1.011 | 1.016 | 1.016 | 1.016 | 1.016 |
| 4096 | 1.049 | 1.075 | 1.075 | 1.075 | 1.076 | 1.076 | 1.076 |
| 8192 | 1.021 | 1.018 | 1.016 | 1.010 | 1.022 | 1.022 | 1.022 |
| 16384 | 1.011 | 1.016 | 1.030 | 0.998 | 1.034 | 1.048 | 1.048 |

</details>

<details><summary>H100, 1600 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 0.992 | 1.009 | 1.012 | 1.058 | 1.103 | 1.131 | 1.165 |
| 2048 | 1.006 | 0.991 | 0.997 | 1.035 | 1.061 | 1.120 | 1.147 |
| 4096 | 0.981 | 0.996 | 0.990 | 1.000 | 1.026 | 1.089 | 1.116 |
| 8192 | 0.998 | 0.968 | 0.987 | 1.010 | 1.050 | 1.106 | 1.100 |
| 16384 | 0.967 | 0.954 | 0.953 | 0.971 | 1.033 | 1.101 | 1.157 |

</details>

<details><summary>H100, 1800 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.012 | 1.013 | 1.003 | 1.030 | 1.092 | 1.120 | 1.157 |
| 2048 | 0.991 | 0.964 | 0.986 | 1.003 | 1.041 | 1.108 | 1.124 |
| 4096 | 0.988 | 0.969 | 0.959 | 1.002 | 1.028 | 1.086 | 1.112 |
| 8192 | 0.990 | 0.972 | 1.005 | 1.016 | 1.060 | 1.108 | 1.108 |
| 16384 | 0.974 | 0.953 | 0.955 | 0.972 | 1.027 | 1.113 | 1.159 |

</details>

<details><summary>H100, 1980 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.001 | 1.017 | 1.020 | 1.046 | 1.101 | 1.135 | 1.163 |
| 2048 | 0.996 | 0.994 | 0.994 | 1.020 | 1.049 | 1.118 | 1.145 |
| 4096 | 0.988 | 0.968 | 0.981 | 1.006 | 1.045 | 1.101 | 1.113 |
| 8192 | 0.975 | 0.987 | 0.999 | 1.006 | 1.046 | 1.107 | 1.108 |
| 16384 | 0.955 | 0.948 | 0.948 | 0.947 | 1.030 | 1.101 | 1.142 |

</details>

<details><summary>B200, free-running</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 0.998 | 1.014 | 1.045 | 1.068 | 1.070 | 1.088 |
| 2048 | 1.001 | 0.998 | 1.004 | 1.013 | 1.036 | 1.052 | 1.078 |
| 4096 | 1.000 | 0.998 | 1.011 | 1.015 | 1.043 | 1.065 | 1.100 |
| 8192 | 0.992 | 1.014 | 1.015 | 1.030 | 1.051 | 1.054 | 1.080 |
| 16384 | 1.003 | 1.001 | 1.007 | 1.019 | 1.059 | 1.062 | 1.049 |

</details>

<details><summary>B200, 1000 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.001 | 1.002 | 1.002 | 1.003 | 1.004 | 1.004 |
| 2048 | 0.999 | 1.000 | 1.001 | 1.003 | 1.005 | 1.006 | 1.007 |
| 4096 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 |
| 8192 | 1.000 | 1.000 | 1.001 | 1.001 | 1.002 | 1.002 | 1.003 |
| 16384 | 0.999 | 1.000 | 1.000 | 1.003 | 1.003 | 1.003 | 1.003 |

</details>

<details><summary>B200, 1200 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.001 | 1.001 | 1.002 | 1.002 | 1.002 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.001 | 1.001 | 1.002 | 1.002 |
| 4096 | 1.019 | 1.026 | 1.026 | 1.026 | 1.026 | 1.026 | 1.026 |
| 8192 | 1.020 | 1.021 | 1.020 | 1.021 | 1.021 | 1.021 | 1.021 |
| 16384 | 1.021 | 1.013 | 1.024 | 1.029 | 1.029 | 1.029 | 1.029 |

</details>

<details><summary>B200, 1400 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.001 | 1.015 | 1.040 | 1.047 | 1.062 | 1.057 | 1.063 |
| 2048 | 1.007 | 1.004 | 1.009 | 1.042 | 1.047 | 1.059 | 1.068 |
| 4096 | 1.017 | 1.019 | 1.020 | 1.036 | 1.088 | 1.072 | 1.067 |
| 8192 | 1.012 | 0.992 | 1.034 | 1.027 | 1.051 | 1.059 | 1.064 |
| 16384 | 1.001 | 0.989 | 1.008 | 1.023 | 1.043 | 1.057 | 1.064 |

</details>

<details><summary>B200, 1600 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.001 | 1.020 | 1.049 | 1.074 | 1.056 | 1.068 |
| 2048 | 0.993 | 0.977 | 0.999 | 1.014 | 1.040 | 1.052 | 1.090 |
| 4096 | 0.998 | 1.005 | 1.021 | 1.035 | 1.060 | 1.058 | 1.069 |
| 8192 | 0.991 | 1.006 | 1.010 | 1.022 | 1.039 | 1.048 | 1.050 |
| 16384 | 0.999 | 1.004 | 1.004 | 1.018 | 1.040 | 1.057 | 1.078 |

</details>

<details><summary>B200, 1800 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.005 | 1.036 | 1.051 | 1.067 | 1.077 | 1.044 |
| 2048 | 0.996 | 0.992 | 1.008 | 1.021 | 1.043 | 1.044 | 1.064 |
| 4096 | 1.003 | 0.992 | 1.018 | 1.027 | 1.068 | 1.065 | 1.082 |
| 8192 | 1.016 | 1.000 | 1.028 | 1.043 | 1.057 | 1.072 | 1.068 |
| 16384 | 0.996 | 0.999 | 1.011 | 1.020 | 1.039 | 1.058 | 1.073 |

</details>

<details><summary>B200, 1965 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 0.994 | 1.000 | 1.003 | 1.010 | 1.047 | 1.050 | 1.081 |
| 2048 | 0.991 | 0.997 | 1.003 | 1.020 | 1.039 | 1.043 | 1.071 |
| 4096 | 0.991 | 1.001 | 1.010 | 1.021 | 1.038 | 1.058 | 1.068 |
| 8192 | 1.007 | 0.994 | 1.022 | 1.027 | 1.054 | 1.060 | 1.071 |
| 16384 | 0.999 | 1.001 | 1.012 | 1.009 | 1.040 | 1.058 | 1.076 |

</details>

Magnitude-pruning results and raw timings, SM clocks and power samples are in [the repo](https://github.com/GeometricAGI/blog/tree/main/sparse-matmul-speedup/results).

### What this means

- **It is a power effect, not a compute effect.** The kernel does the same work. It only runs faster when the GPU is power-limited and the zeros let it hold a higher clock; at a clock the GPU can sustain, the time is identical.
- **Benchmark with care.** If you compare kernels or models and one of them runs on weights with many zeros (pruned, quantized, or just initialized differently), part of the gap can be the power cap rather than the kernel. Lock clocks to a value below the power-limited clock, or at least log SM clock and power next to your timings.
- **Don't count on it.** The effect is small, only shows up for compute-bound shapes at power-limited clocks, and is smaller than the noise at the sparsity levels that are common in practice.

Limitations: one weight shape (8192x8192), bf16 only, one GPU of each type, and power readings from NVML are sampled, not integrated. A natural follow-up is to compare against a genuine sparse kernel path such as cuSPARSELt, and to repeat this for fp8.
