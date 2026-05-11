"""Compute geometric-mean speedups and generate plots for the HF kernel-hub blog.

Reads the in-house benchmark CSVs in $BENCH_DIR (one per kernel-vs-baseline
comparison), filters to the cute_* rows that have a precomputed
LATENCY_SPEEDUP, and emits:

  * a geomean summary printed to stdout, formatted to drop directly into the
    blog post tables.
  * a per-kernel grouped bar chart (vs eager and vs torch.compile
    max-autotune-no-cudagraphs) saved under $OUT_DIR.

For grpo/bnpo the recorded shape is (B, S) but the kernels are called with a
group dimension of 8, so the effective batch is 8*B. Reverse-KL has no group
multiplier.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BENCH_DIR = Path(
    os.environ.get("BENCH_DIR", "/home/pramodith/geo-evo/benchmark_results")
)
OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        "/home/pramodith/GeometricAGI.github.io/assets/hf-kernel-hub",
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# (group_label, blog_kernel_name, cute_kernel_in_csv, csv_stem, group_multiplier)
SECTIONS = [
    ("GRPO beta!=0", "grpo_loss_fwd", "cute_grpo_loss_mask",
     "grpo_loss_mask", 8),
    ("GRPO beta!=0", "grpo_loss", "cute_grpo_loss_fwd_bwd_mask",
     "grpo_loss_fwd_bwd_mask", 8),
    ("GRPO beta==0", "grpo_loss_fwd", "cute_grpo_loss_no_kl_mask",
     "grpo_loss_no_kl_mask", 8),
    ("GRPO beta==0", "grpo_loss", "cute_grpo_loss_fwd_bwd_no_kl_mask",
     "grpo_loss_fwd_bwd_no_kl_mask", 8),
    ("BNPO beta!=0", "bnpo_loss_fwd", "cute_bnpo_loss",
     "bnpo_loss", 8),
    ("BNPO beta!=0", "bnpo_loss", "cute_bnpo_loss_fwd_bwd",
     "bnpo_loss_fwd_bwd", 8),
    ("BNPO beta==0", "bnpo_loss_fwd", "cute_bnpo_loss_no_kl",
     "bnpo_loss_no_kl", 8),
    ("BNPO beta==0", "bnpo_loss", "cute_bnpo_loss_fwd_bwd_no_kl",
     "bnpo_loss_fwd_bwd_no_kl", 8),
    ("Reverse KL", "reverse_kl_fwd", "cute_reverse_kl_div",
     "reverse_kl_div", 1),
    ("Reverse KL", "reverse_kl_loss", "cute_reverse_kl_div_fwd_bwd",
     "reverse_kl_div_fwd_bwd", 1),
]


def parse_shape(raw: str) -> tuple[int, int]:
    raw = raw.strip().strip('"').strip("()")
    a, b = raw.split(",")
    return int(a.strip()), int(b.strip())


def shape_label(shape: tuple[int, int], mult: int) -> str:
    return f"({shape[0] * mult}, {shape[1]})"


def load_speedups(csv_path: Path, cute_kernel: str) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["kernel"] != cute_kernel:
                continue
            sp = row.get("LATENCY_SPEEDUP", "").strip()
            if not sp:
                continue
            try:
                out[parse_shape(row["shape"])] = float(sp)
            except ValueError:
                pass
    return out


def load_latencies(csv_path: Path) -> dict[str, dict[tuple[int, int], float]]:
    """Return {kernel_name: {shape: median_latency_ms}} for all kernels in the CSV."""
    out: dict[str, dict[tuple[int, int], float]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            med = row.get("LATENCY_MEDIAN", "").strip()
            if not med:
                continue
            try:
                lat = float(med)
            except ValueError:
                continue
            shape = parse_shape(row["shape"])
            out.setdefault(row["kernel"], {})[shape] = lat
    return out


def geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _draw_latency_subplot(ax, shapes, mult, cute_lat, base_lat, baseline_label,
                          baseline_color, speedup_map):
    labels = [shape_label(sh, mult) for sh in shapes]
    cute_vals = [cute_lat.get(sh, np.nan) for sh in shapes]
    base_vals = [base_lat.get(sh, np.nan) for sh in shapes]
    y = np.arange(len(shapes))
    height = 0.38
    ax.barh(y + height / 2, base_vals, height, label=baseline_label,
            color=baseline_color)
    b_cute = ax.barh(y - height / 2, cute_vals, height, label="cute",
                     color="#54A24B")
    for bar, sh in zip(b_cute, shapes):
        sp = speedup_map.get(sh)
        if sp is not None and not math.isnan(bar.get_width()):
            ax.text(bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"  {sp:.2f}x", ha="left", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Median latency (ms) — shorter is better")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xscale("log")
    ax.grid(True, axis="x", which="both", linestyle=":", linewidth=0.5,
            alpha=0.5)


def plot_kernel(group, kernel, mult, cute_kernel, csv_stem, out_path):
    compile_csv = BENCH_DIR / f"{csv_stem}_comparison_prefill.csv"
    eager_csv = BENCH_DIR / f"{csv_stem}_eager_comparison_prefill.csv"

    compile_lats = load_latencies(compile_csv)
    eager_lats = load_latencies(eager_csv)
    compile_speedups = load_speedups(compile_csv, cute_kernel)
    eager_speedups = load_speedups(eager_csv, cute_kernel)

    cute_compile = compile_lats.get(cute_kernel, {})
    cute_eager = eager_lats.get(cute_kernel, {})
    torch_compile_kernel = next((k for k in compile_lats
                                 if k.startswith("torch_") and "eager" not in k),
                                None)
    torch_eager_kernel = next((k for k in eager_lats if k.endswith("_eager")), None)
    base_compile = compile_lats.get(torch_compile_kernel, {}) if torch_compile_kernel else {}
    base_eager = eager_lats.get(torch_eager_kernel, {}) if torch_eager_kernel else {}

    compile_shapes = sorted(set(cute_compile) & set(base_compile))
    eager_shapes = sorted(set(cute_eager) & set(base_eager))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].set_title(f"vs torch.compile (max-autotune-no-cudagraphs)")
    _draw_latency_subplot(axes[0], compile_shapes, mult, cute_compile,
                          base_compile, "torch.compile", "#F58518",
                          compile_speedups)
    axes[1].set_title("vs Eager")
    _draw_latency_subplot(axes[1], eager_shapes, mult, cute_eager,
                          base_eager, "eager", "#4C78A8", eager_speedups)
    fig.suptitle(f"{group} — {kernel}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    summaries = []
    for group, kernel, cute_name, stem, mult in SECTIONS:
        compile_csv = BENCH_DIR / f"{stem}_comparison_prefill.csv"
        eager_csv = BENCH_DIR / f"{stem}_eager_comparison_prefill.csv"
        compile_map = load_speedups(compile_csv, cute_name)
        eager_map = load_speedups(eager_csv, cute_name)
        shapes = sorted(set(compile_map) | set(eager_map))

        labels = [shape_label(sh, mult) for sh in shapes]
        eager_vals = [eager_map.get(sh) for sh in shapes]
        compile_vals = [compile_map.get(sh) for sh in shapes]
        gm_eager = geomean([v for v in eager_vals if v is not None])
        gm_compile = geomean([v for v in compile_vals if v is not None])
        summaries.append({
            "group": group, "kernel": kernel, "stem": stem,
            "labels": labels, "eager": eager_vals, "compile": compile_vals,
            "gm_eager": gm_eager, "gm_compile": gm_compile,
        })
        plot_kernel(
            group, kernel, mult, cute_name, stem,
            OUT_DIR / f"{stem}_speedup.png",
        )

    print("=" * 90)
    print("Geometric-mean speedups (n = number of shapes contributing)")
    print("=" * 90)
    print(f"{'group':<14}{'kernel':<22}{'gm vs eager':>14}{'gm vs compile':>16}{'  n_eager  n_compile'}")
    for s in summaries:
        gm_e = f"{s['gm_eager']:.2f}x" if s["gm_eager"] else "n/a"
        gm_c = f"{s['gm_compile']:.2f}x" if s["gm_compile"] else "n/a"
        n_e = sum(v is not None for v in s["eager"])
        n_c = sum(v is not None for v in s["compile"])
        print(f"{s['group']:<14}{s['kernel']:<22}{gm_e:>14}{gm_c:>16}{n_e:>10}{n_c:>11}")

    print()
    for s in summaries:
        print(f"\n--- {s['group']} / {s['kernel']}  ({s['stem']}) ---")
        print(f"{'shape':<16}{'vs eager':>14}{'vs compile':>16}")
        for label, e, c in zip(s["labels"], s["eager"], s["compile"]):
            es = f"{e:.3f}x" if e is not None else "-"
            cs = f"{c:.3f}x" if c is not None else "-"
            print(f"{label:<16}{es:>14}{cs:>16}")

    print(f"\nPlots written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
