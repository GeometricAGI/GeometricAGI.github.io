"""Summarize the Hugging Face `kernels`-style benchmark results.

Reads the per-suite results.json under
$KBENCH_DIR/{suite}/results.json (suites: bnpo_loss_{compiled,eager},
grpo_loss_{compiled,eager}, reverse_kl_{compiled,eager}). For every workload,
speedup = refMeanMs / mean_ms; we then group by (kernel-family, variant)
and report per-shape speedups plus the geometric mean. Also copies the
dark-themed *_latency.svg plots that ship next to each results.json into
$OUT_DIR for embedding in the blog post.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

KBENCH_DIR = Path(
    "/home/pramodith/geometric-hfhub-kernels/benchmark_results"
)
OUT_DIR = Path(
    "/home/pramodith/GeometricAGI.github.io/assets/hf-kernel-hub"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUITES = [
    ("grpo", "compiled"),
    ("grpo", "eager"),
    ("bnpo", "compiled"),
    ("bnpo", "eager"),
    ("reverse_kl", "compiled"),
    ("reverse_kl", "eager"),
]

# Map suite-prefix -> directory name
DIR_FOR = {
    ("grpo", "compiled"): "grpo_loss_compiled",
    ("grpo", "eager"): "grpo_loss_eager",
    ("bnpo", "compiled"): "bnpo_loss_compiled",
    ("bnpo", "eager"): "bnpo_loss_eager",
    ("reverse_kl", "compiled"): "reverse_kl_compiled",
    ("reverse_kl", "eager"): "reverse_kl_eager",
}

WORKLOAD_RE = re.compile(
    r"\.(?P<kernel>[a-z_]+?_loss(?:_fwd)?|reverse_kl(?:_fwd)?)_"
    r"batch(?P<batch>\d+)_seqlen(?P<seqlen>\d+)"
    r"(?:_vocab(?P<vocab>\d+))?_(?:compiled|eager)$"
)


def parse_workload(name: str) -> tuple[str, tuple[int, int]] | None:
    m = WORKLOAD_RE.search(name)
    if not m:
        return None
    return m.group("kernel"), (int(m.group("batch")), int(m.group("seqlen")))


def geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load_suite(family: str, baseline: str) -> dict[str, dict[tuple[int, int], float]]:
    """Return {kernel_variant: {shape: speedup}}."""
    path = KBENCH_DIR / DIR_FOR[(family, baseline)] / "results.json"
    data = json.loads(path.read_text())
    out: dict[str, dict[tuple[int, int], float]] = {}
    for r in data["results"]:
        parsed = parse_workload(r["workload"])
        if parsed is None:
            continue
        kernel, shape = parsed
        t = r["timingResults"]
        if not t.get("mean_ms") or not t.get("refMeanMs"):
            continue
        speedup = t["refMeanMs"] / t["mean_ms"]
        out.setdefault(kernel, {})[shape] = speedup
    return out


def copy_dark_plots() -> list[tuple[str, Path]]:
    copied = []
    for (family, baseline), dirname in DIR_FOR.items():
        src = KBENCH_DIR / dirname / f"{dirname}_dark_latency.svg"
        if not src.exists():
            print(f"warn: {src} missing")
            continue
        dst = OUT_DIR / f"{dirname}_dark_latency.svg"
        shutil.copyfile(src, dst)
        copied.append((dirname, dst))
    return copied


def main() -> None:
    rows = []
    for family in ("grpo", "bnpo", "reverse_kl"):
        compiled = load_suite(family, "compiled")
        eager = load_suite(family, "eager")
        kernels = sorted(set(compiled) | set(eager))
        for k in kernels:
            c = compiled.get(k, {})
            e = eager.get(k, {})
            shapes = sorted(set(c) | set(e))
            rows.append({
                "family": family,
                "kernel": k,
                "shapes": shapes,
                "compiled": c,
                "eager": e,
                "gm_compiled": geomean(list(c.values())),
                "gm_eager": geomean(list(e.values())),
            })

    print("=" * 90)
    print("HF kernels-style speedups (refMeanMs / mean_ms)")
    print("=" * 90)
    print(f"{'family':<12}{'kernel':<22}{'gm vs eager':>14}{'gm vs compiled':>16}{'  n_eager  n_compiled'}")
    for r in rows:
        gm_e = f"{r['gm_eager']:.2f}x" if r["gm_eager"] else "n/a"
        gm_c = f"{r['gm_compiled']:.2f}x" if r["gm_compiled"] else "n/a"
        print(f"{r['family']:<12}{r['kernel']:<22}{gm_e:>14}{gm_c:>16}{len(r['eager']):>10}{len(r['compiled']):>11}")

    print()
    for r in rows:
        print(f"\n--- {r['family']} / {r['kernel']} ---")
        print(f"{'shape':<18}{'vs eager':>14}{'vs compiled':>16}")
        for sh in r["shapes"]:
            es = f"{r['eager'][sh]:.3f}x" if sh in r["eager"] else "-"
            cs = f"{r['compiled'][sh]:.3f}x" if sh in r["compiled"] else "-"
            print(f"{str(sh):<18}{es:>14}{cs:>16}")

    print("\nCopying dark .svg latency plots to", OUT_DIR)
    for name, dst in copy_dark_plots():
        print(f"  {name}_dark_latency.svg -> {dst.name}")


if __name__ == "__main__":
    main()
