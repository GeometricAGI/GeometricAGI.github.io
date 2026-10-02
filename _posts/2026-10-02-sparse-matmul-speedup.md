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
- We set exactly that fraction of `w` to zero (5%, 10%, 25%, 50%, 70%, 90%, 99%), at positions chosen uniformly at random; the measured zero fraction matches to within 2e-7 and is stored in the results. Magnitude pruning (zeroing the smallest-magnitude weights) is also in the repo. It agrees with random zeroing at 99% zeros, but at 5% zeros it is faster than random zeroing on the B200 (1.02-1.10x vs. about 1.0x), which we have not explained, so we only tabulate random zeroing here. Everything below is "speedup over the dense weight", i.e. dense time divided by time with zeros, so values above 1.000 mean faster than dense.
- Matmuls are captured in a CUDA graph and timed over replays. Each graph rotates through 8 independent copies of the weight, the activations and the output (each weight copy has its own random zero positions), so the same tensor is only touched again after at least a gigabyte of other traffic, far more than the L2 cache. Without this, the same 128 MiB weight would be re-read from cache on every replay. Every config is timed in 5 interleaved rounds (alternating order) and we report the median, so drift hits every config equally.
- `M` is swept over 1024, 2048, 4096, 8192 and 16384. In an exploratory run on the H100 with smaller `M` (1, 16 and 128 rows) the matmul is memory-bound and sparsity made no difference (within 0.5%), so we start the published sweep at 1024.
- Each GPU is run with free-running clocks and with the SM clock locked (`nvidia-smi -lgc`) at six values: 1000, 1200, 1400, 1600, 1800 MHz and the GPU's maximum (1980 MHz on the H100, 1965 MHz on the B200).

### Free-running clocks: dense matmuls do get faster

Speedup with 99% of the weights zeroed, free-running clocks:

| | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| H100 | 1.125 | 1.111 | 1.107 | 1.148 | 1.175 |
| B200 | 1.098 | 1.078 | 1.117 | 1.103 | 1.104 |

![H100, free-running clocks](/assets/sparse-matmul-speedup/h100-unlocked.png)

![B200, free-running clocks](/assets/sparse-matmul-speedup/b200-unlocked.png)

- Speedup generally grows with the amount of zeros, up to about **1.18x on the H100 and 1.12x on the B200**.
- At realistic sparsity (5-25%) the effect is mostly inside the run-to-run noise. On the H100, 5-25% zeros was up to 2% **slower** than dense for several shapes.
- Differences of a couple of percent between neighbouring points are within run-to-run noise.

### Locked clocks: the speedup depends on the lock

Speedup with 99% zeros for each locked SM clock (the first row is the free-running result for reference):

**H100**

| SM clock | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| free-running | 1.125 | 1.111 | 1.107 | 1.148 | 1.175 |
| 1000 MHz | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 |
| 1200 MHz | 1.000 | 1.000 | 1.000 | 1.002 | 1.004 |
| 1400 MHz | 1.001 | 1.000 | 1.022 | 1.016 | 1.012 |
| 1600 MHz | 1.133 | 1.109 | 1.116 | 1.159 | 1.195 |
| 1800 MHz | 1.137 | 1.111 | 1.121 | 1.143 | 1.180 |
| 1980 MHz | 1.134 | 1.097 | 1.123 | 1.144 | 1.186 |

**B200**

| SM clock | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|
| free-running | 1.098 | 1.078 | 1.117 | 1.103 | 1.104 |
| 1000 MHz | 1.004 | 1.003 | 1.001 | 1.001 | 0.995 |
| 1200 MHz | 1.003 | 1.001 | 1.016 | 1.016 | 1.021 |
| 1400 MHz | 1.060 | 1.078 | 1.110 | 1.108 | 1.107 |
| 1600 MHz | 1.099 | 1.059 | 1.098 | 1.113 | 1.097 |
| 1800 MHz | 1.077 | 1.083 | 1.114 | 1.108 | 1.106 |
| 1965 MHz | 1.109 | 1.104 | 1.102 | 1.107 | 1.100 |

![H100, speedup vs. locked clock](/assets/sparse-matmul-speedup/h100-clock_sweep.png)

![B200, speedup vs. locked clock](/assets/sparse-matmul-speedup/b200-clock_sweep.png)

- **Low locks: no effect.** At 1000 and 1200 MHz the H100 is within 0.4% of dense for every `M` and every sparsity level, and the B200 is within 0.6% at 1000 MHz. The GPU holds the requested clock, so the work takes the same time whatever is in the weights.
- **High locks: the speedup comes back.** On the H100 it is mixed at 1400 MHz (0.92-1.02x) and fully back, at its free-running size, from 1600 MHz up. On the B200 it is back from 1400 MHz, and partly there already at 1200 MHz (up to 1.02x for the larger shapes). Above that the lock stops being the binding constraint: dense runs reach only 1320-1410 MHz on the H100 and 1185-1320 MHz on the B200 no matter how high we set the lock, because the power limit decides the clock. Sparser weights draw less power, so they sustain a somewhat higher clock.
- The right-hand panels plot the same speedups against the clock the dense run actually reached (a single NVML sample per run, so noisy). The 1600-1980 MHz locks all land on top of each other at about 1320-1400 MHz, which is why they give the same speedup. The 1400 MHz lock reaches a similar clock yet shows little speedup, so the sampled clock alone does not predict the effect; what matters is whether the power limit is binding.
- A clock lock is therefore only a control if it is set below the clock the GPU would sustain at its power limit for that workload.

### Full results

The tables below give the speedup for every `M` and sparsity level for each clock setting.

<details><summary>H100, free-running</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 0.998 | 0.983 | 0.983 | 1.015 | 1.048 | 1.103 | 1.125 |
| 2048 | 0.989 | 0.986 | 0.985 | 1.002 | 1.039 | 1.090 | 1.111 |
| 4096 | 0.999 | 0.989 | 1.003 | 1.032 | 1.068 | 1.108 | 1.107 |
| 8192 | 1.036 | 1.031 | 1.033 | 1.059 | 1.097 | 1.123 | 1.148 |
| 16384 | 0.994 | 0.973 | 0.978 | 1.009 | 1.048 | 1.135 | 1.175 |

</details>

<details><summary>H100, 1000 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.001 | 1.001 |
| 16384 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

</details>

<details><summary>H100, 1200 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 0.998 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 1.000 |
| 2048 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4096 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8192 | 1.000 | 1.002 | 1.000 | 1.002 | 1.002 | 1.003 | 1.002 |
| 16384 | 1.003 | 1.002 | 0.999 | 0.999 | 1.004 | 1.004 | 1.004 |

</details>

<details><summary>H100, 1400 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.001 | 1.001 | 1.001 | 1.002 | 1.001 |
| 2048 | 1.000 | 1.000 | 1.001 | 1.001 | 1.001 | 1.001 | 1.000 |
| 4096 | 1.010 | 1.018 | 1.022 | 1.020 | 1.022 | 1.022 | 1.022 |
| 8192 | 1.008 | 1.013 | 1.014 | 1.014 | 1.016 | 1.015 | 1.016 |
| 16384 | 0.975 | 0.916 | 0.947 | 1.002 | 1.009 | 1.011 | 1.012 |

</details>

<details><summary>H100, 1600 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.002 | 1.005 | 1.007 | 1.043 | 1.059 | 1.123 | 1.133 |
| 2048 | 0.988 | 0.978 | 0.988 | 1.004 | 1.028 | 1.096 | 1.109 |
| 4096 | 0.999 | 0.983 | 1.001 | 1.032 | 1.074 | 1.106 | 1.116 |
| 8192 | 1.026 | 1.022 | 1.026 | 1.046 | 1.093 | 1.128 | 1.159 |
| 16384 | 1.017 | 1.008 | 0.993 | 1.025 | 1.082 | 1.162 | 1.195 |

</details>

<details><summary>H100, 1800 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.001 | 1.001 | 1.008 | 1.062 | 1.123 | 1.137 |
| 2048 | 0.992 | 0.992 | 0.984 | 1.016 | 1.042 | 1.100 | 1.111 |
| 4096 | 0.989 | 0.996 | 0.994 | 1.041 | 1.068 | 1.108 | 1.121 |
| 8192 | 1.021 | 1.017 | 1.021 | 1.044 | 1.077 | 1.120 | 1.143 |
| 16384 | 0.999 | 0.988 | 0.974 | 1.006 | 1.050 | 1.142 | 1.180 |

</details>

<details><summary>H100, 1980 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 0.999 | 1.009 | 1.040 | 1.063 | 1.117 | 1.134 |
| 2048 | 0.990 | 0.990 | 0.990 | 1.000 | 1.047 | 1.093 | 1.097 |
| 4096 | 0.998 | 0.986 | 1.003 | 1.045 | 1.066 | 1.104 | 1.123 |
| 8192 | 1.013 | 1.020 | 1.015 | 1.046 | 1.064 | 1.122 | 1.144 |
| 16384 | 1.015 | 0.992 | 0.984 | 1.016 | 1.070 | 1.148 | 1.186 |

</details>

<details><summary>B200, free-running</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.006 | 1.011 | 1.054 | 1.074 | 1.071 | 1.098 |
| 2048 | 0.978 | 0.997 | 0.989 | 1.023 | 1.037 | 1.062 | 1.078 |
| 4096 | 0.994 | 0.993 | 1.013 | 1.044 | 1.063 | 1.074 | 1.117 |
| 8192 | 1.003 | 1.003 | 1.022 | 1.036 | 1.060 | 1.076 | 1.103 |
| 16384 | 1.000 | 1.000 | 1.008 | 1.035 | 1.070 | 1.086 | 1.104 |

</details>

<details><summary>B200, 1000 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.001 | 1.002 | 1.003 | 1.005 | 1.004 |
| 2048 | 1.001 | 1.000 | 1.001 | 1.001 | 1.002 | 1.003 | 1.003 |
| 4096 | 1.000 | 0.999 | 1.001 | 1.001 | 1.001 | 1.000 | 1.001 |
| 8192 | 1.000 | 1.000 | 1.001 | 1.001 | 1.002 | 1.002 | 1.001 |
| 16384 | 1.000 | 1.000 | 0.998 | 0.994 | 0.995 | 0.995 | 0.995 |

</details>

<details><summary>B200, 1200 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.000 | 1.001 | 1.002 | 1.002 | 1.003 | 1.003 |
| 2048 | 1.000 | 1.000 | 1.000 | 1.001 | 1.001 | 1.001 | 1.001 |
| 4096 | 0.993 | 1.007 | 1.014 | 1.015 | 1.015 | 1.015 | 1.016 |
| 8192 | 0.989 | 0.992 | 1.015 | 1.015 | 1.016 | 1.016 | 1.016 |
| 16384 | 1.004 | 1.002 | 1.008 | 1.021 | 1.021 | 1.021 | 1.021 |

</details>

<details><summary>B200, 1400 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 1.002 | 1.022 | 1.046 | 1.057 | 1.059 | 1.060 |
| 2048 | 1.006 | 0.996 | 1.002 | 1.027 | 1.059 | 1.065 | 1.078 |
| 4096 | 0.997 | 0.994 | 1.015 | 1.033 | 1.056 | 1.073 | 1.110 |
| 8192 | 0.987 | 1.005 | 1.011 | 1.034 | 1.060 | 1.079 | 1.108 |
| 16384 | 1.004 | 1.003 | 1.012 | 1.034 | 1.062 | 1.096 | 1.107 |

</details>

<details><summary>B200, 1600 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.001 | 1.002 | 1.029 | 1.046 | 1.070 | 1.073 | 1.099 |
| 2048 | 1.001 | 1.003 | 1.013 | 1.030 | 1.045 | 1.058 | 1.059 |
| 4096 | 0.996 | 0.987 | 0.997 | 1.031 | 1.049 | 1.071 | 1.098 |
| 8192 | 1.009 | 0.996 | 1.026 | 1.043 | 1.072 | 1.105 | 1.113 |
| 16384 | 1.001 | 1.001 | 1.006 | 1.034 | 1.058 | 1.093 | 1.097 |

</details>

<details><summary>B200, 1800 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.001 | 1.002 | 1.015 | 1.043 | 1.078 | 1.037 | 1.077 |
| 2048 | 0.987 | 0.986 | 0.990 | 1.009 | 1.041 | 1.052 | 1.083 |
| 4096 | 0.994 | 1.000 | 1.004 | 1.042 | 1.057 | 1.074 | 1.114 |
| 8192 | 0.987 | 1.003 | 1.017 | 1.038 | 1.059 | 1.095 | 1.108 |
| 16384 | 1.003 | 1.003 | 1.011 | 1.034 | 1.061 | 1.095 | 1.106 |

</details>

<details><summary>B200, 1965 MHz</summary>

| M | 5% | 10% | 25% | 50% | 70% | 90% | 99% |
|---|---|---|---|---|---|---|---|
| 1024 | 1.000 | 0.996 | 1.005 | 1.043 | 1.067 | 1.054 | 1.109 |
| 2048 | 1.009 | 1.000 | 1.017 | 1.029 | 1.050 | 1.074 | 1.104 |
| 4096 | 1.000 | 0.993 | 1.009 | 1.034 | 1.054 | 1.074 | 1.102 |
| 8192 | 0.997 | 1.011 | 1.011 | 1.032 | 1.057 | 1.093 | 1.107 |
| 16384 | 1.000 | 0.999 | 1.004 | 1.029 | 1.059 | 1.090 | 1.100 |

</details>

Magnitude-pruning results and raw timings, SM clocks and power samples are in [the repo](https://github.com/GeometricAGI/blog/tree/main/sparse-matmul-speedup/results).

### Realistic LLM shapes: DeepSeek-V4-Flash

The square 8192x8192 matmul above is a clean test, but not what an inference engine runs. So we repeated the experiment with the weight shapes of [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (hidden size 4096, 64 heads of dim 512, 256 routed experts of intermediate size 2048, 6 experts per token, vocabulary 129,280) and with token counts from single-token decode up to an 8192-token prefill chunk:

- `q_b_proj`: 1024 to 32768 (the query up-projection).
- `expert_gate_up`: 4096 to 4096 (one routed expert's fused gate and up projection). With 6 of 256 experts per token, an expert sees about `tokens x 6/256` rows, so we scale the row count that way (for example 512 tokens means 12 rows).
- `lm_head`: 4096 to 129280.

Weights are stored as `(out, in)` and applied as `x @ w.t()`, as in `nn.Linear`, in bf16 (the released model uses fp8 and fp4 weights, which we did not test). Everything else is as before, with each graph rotating through enough weight copies to cover at least 512 MiB. Each GPU is run free-running and with one low locked clock, 1200 MHz on the H100 and 1000 MHz on the B200. Speedup over the dense weight for 50% and 99% random zeros:

**H100**

| Weight | Tokens (rows) | dense time (us) | free-running 50% | free-running 99% | locked 1200 MHz 50% | locked 1200 MHz 99% |
|---|---|---|---|---|---|---|
| q_b_proj (1024 to 32768) | 1 (1) | 24.4 | 0.999 | 1.003 | 0.998 | 1.008 |
| q_b_proj (1024 to 32768) | 8 (8) | 24.9 | 0.999 | 1.002 | 0.998 | 1.008 |
| q_b_proj (1024 to 32768) | 32 (32) | 25.7 | 0.998 | 1.003 | 0.999 | 1.009 |
| q_b_proj (1024 to 32768) | 128 (128) | 29.7 | 1.000 | 1.003 | 0.995 | 1.007 |
| q_b_proj (1024 to 32768) | 512 (512) | 62.2 | 1.042 | 1.166 | 1.001 | 1.007 |
| q_b_proj (1024 to 32768) | 2048 (2048) | 220.7 | 0.991 | 1.085 | 1.001 | 1.002 |
| q_b_proj (1024 to 32768) | 8192 (8192) | 872.0 | 1.035 | 1.146 | 1.000 | 1.000 |
| expert_gate_up (4096 to 4096, routed) | 1 (1) | 16.0 | 1.002 | 1.002 | 1.001 | 1.017 |
| expert_gate_up (4096 to 4096, routed) | 128 (3) | 16.0 | 1.000 | 1.000 | 1.000 | 1.016 |
| expert_gate_up (4096 to 4096, routed) | 512 (12) | 16.2 | 1.000 | 0.999 | 1.002 | 1.010 |
| expert_gate_up (4096 to 4096, routed) | 2048 (48) | 14.5 | 0.995 | 0.997 | 0.996 | 1.010 |
| expert_gate_up (4096 to 4096, routed) | 8192 (192) | 16.7 | 1.007 | 1.023 | 1.001 | 1.005 |
| lm_head (4096 to 129280) | 1 (1) | 361.0 | 1.000 | 1.003 | 0.996 | 1.004 |
| lm_head (4096 to 129280) | 8 (8) | 364.0 | 1.002 | 1.001 | 0.997 | 1.003 |
| lm_head (4096 to 129280) | 32 (32) | 367.0 | 1.003 | 0.998 | 1.003 | 1.002 |
| lm_head (4096 to 129280) | 128 (128) | 378.9 | 0.998 | 1.004 | 0.999 | 0.997 |
| lm_head (4096 to 129280) | 512 (512) | 867.5 | 1.023 | 1.129 | 1.002 | 1.003 |
| lm_head (4096 to 129280) | 2048 (2048) | 3173.9 | 1.040 | 1.152 | 1.000 | 1.000 |
| lm_head (4096 to 129280) | 8192 (8192) | 12636.5 | 1.023 | 1.211 | 1.000 | 1.000 |

**B200**

| Weight | Tokens (rows) | dense time (us) | free-running 50% | free-running 99% | locked 1000 MHz 50% | locked 1000 MHz 99% |
|---|---|---|---|---|---|---|
| q_b_proj (1024 to 32768) | 1 (1) | 12.4 | 0.999 | 0.949 | 1.000 | 0.947 |
| q_b_proj (1024 to 32768) | 8 (8) | 12.5 | 1.006 | 0.944 | 1.004 | 0.949 |
| q_b_proj (1024 to 32768) | 32 (32) | 13.2 | 1.005 | 0.950 | 1.002 | 0.951 |
| q_b_proj (1024 to 32768) | 128 (128) | 15.8 | 1.017 | 0.975 | 1.007 | 0.981 |
| q_b_proj (1024 to 32768) | 512 (512) | 30.8 | 1.020 | 1.087 | 0.998 | 1.006 |
| q_b_proj (1024 to 32768) | 2048 (2048) | 101.6 | 1.038 | 1.097 | 1.002 | 1.005 |
| q_b_proj (1024 to 32768) | 8192 (8192) | 396.1 | 1.031 | 1.094 | 1.000 | 1.000 |
| expert_gate_up (4096 to 4096, routed) | 1 (1) | 8.6 | 1.032 | 0.995 | 1.006 | 1.011 |
| expert_gate_up (4096 to 4096, routed) | 128 (3) | 9.1 | 1.001 | 1.001 | 1.001 | 1.007 |
| expert_gate_up (4096 to 4096, routed) | 512 (12) | 8.8 | 1.031 | 0.998 | 1.005 | 1.014 |
| expert_gate_up (4096 to 4096, routed) | 2048 (48) | 9.2 | 1.030 | 0.997 | 1.007 | 1.016 |
| expert_gate_up (4096 to 4096, routed) | 8192 (192) | 12.3 | 1.020 | 1.057 | 0.995 | 0.998 |
| lm_head (4096 to 129280) | 1 (1) | 155.7 | 1.000 | 1.003 | 0.997 | 0.999 |
| lm_head (4096 to 129280) | 8 (8) | 154.5 | 1.000 | 1.004 | 1.000 | 1.001 |
| lm_head (4096 to 129280) | 32 (32) | 162.7 | 1.004 | 1.004 | 1.001 | 1.002 |
| lm_head (4096 to 129280) | 128 (128) | 198.8 | 1.025 | 1.060 | 0.996 | 0.998 |
| lm_head (4096 to 129280) | 512 (512) | 402.9 | 1.036 | 1.094 | 1.000 | 1.001 |
| lm_head (4096 to 129280) | 2048 (2048) | 1453.4 | 1.032 | 1.092 | 1.000 | 1.000 |
| lm_head (4096 to 129280) | 8192 (8192) | 5766.3 | 1.044 | 1.106 | 1.000 | 1.000 |

![H100, DeepSeek-V4-Flash shapes, free-running clocks](/assets/sparse-matmul-speedup/h100-dsv4-flash-unlocked.png)

![B200, DeepSeek-V4-Flash shapes, free-running clocks](/assets/sparse-matmul-speedup/b200-dsv4-flash-unlocked.png)

- **Decode batches (up to 128 tokens): no effect on the H100.** These matmuls are limited by reading the weights from memory and run at the full clock, so the zeros change nothing (within about 1%).
- **Prefill-sized batches: the effect is as large as for the square case, or larger.** `lm_head` with 8192 tokens is up to **1.21x faster on the H100** at 99% zeros and 1.11x on the B200; `q_b_proj` is up to 1.17x and 1.10x. These are the shapes where the dense matmul runs at the power limit and throttles to 1335-1440 MHz.
- **Routed experts barely benefit.** With only up to 192 rows per expert the matmul is short and mostly memory-bound, and the speedup is at most 2% on the H100 and 6% on the B200, and not monotonic in sparsity on the B200.
- **Locking the clock again removes the compute-bound speedup:** at 1200 MHz (H100) and 1000 MHz (B200) the `q_b_proj` and `lm_head` prefill speedups drop to within 1% of 1.0.
- **A B200 surprise we cannot explain.** `q_b_proj` at decode sizes (1 to 128 tokens) is 2-6% *slower* with 90-99% zeros, both free-running and at the locked 1000 MHz clock, so this is not a power or clock effect. It does not appear on the H100, and we have not found the cause.
- A few small residuals (about 1-1.7%) remain for the routed-expert shape at the locked clocks, comparable to the run-to-run noise.

### What this means

- **It is a power effect, not a compute effect.** The kernel does the same work. It only runs faster when the GPU is power-limited and the zeros let it hold a higher clock; at a clock the GPU can sustain, the time is identical.
- **Benchmark with care.** If you compare kernels or models and one of them runs on weights with many zeros (pruned, quantized, or just initialized differently), part of the gap can be the power cap rather than the kernel. Lock clocks to a value below the power-limited clock, or at least log SM clock and power next to your timings.
- **Don't count on it.** The effect is small, only shows up for compute-bound shapes at power-limited clocks, and is smaller than the noise at the sparsity levels that are common in practice.

Limitations: bf16 only, random Gaussian weights rather than trained ones, only three DeepSeek-V4-Flash weight shapes, one GPU of each type, and power readings from NVML are sampled, not integrated. A natural follow-up is to compare against a genuine sparse kernel path such as cuSPARSELt, and to repeat this for fp8.
