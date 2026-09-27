"""PNG charts built from ``metrics.jsonl`` (works while training is still running)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from .tracking import read_metrics

log = logging.getLogger(__name__)

# Reference categorical palette (slot order is fixed): 1 blue, 2 orange, 3 aqua.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """(steps, values) for one metric; for repeated steps (after a resume) the latest wins."""
    by_step: dict[int, float] = {}
    for r in rows:
        v = r.get(key)
        if v is not None and math.isfinite(v):
            by_step[int(r["step"])] = float(v)
    steps = sorted(by_step)
    return steps, [by_step[s] for s in steps]


def _ema(values: list[float], alpha: float = 0.1) -> list[float]:
    out, cur = [], None
    for v in values:
        cur = v if cur is None else alpha * v + (1 - alpha) * cur
        out.append(cur)
    return out


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, fontweight="bold")
    ax.set_xlabel("optimizer step", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _plot_loss(ax, rows) -> bool:
    ts, tv = _series(rows, "train/loss")
    es, ev = _series(rows, "eval/loss")
    if not ts and not es:
        return False
    if ts:
        ax.plot(ts, tv, color=SERIES[0], alpha=0.25, linewidth=1)
        ax.plot(ts, _ema(tv), color=SERIES[0], linewidth=2, label="train loss (smoothed)")
    if es:
        ax.plot(es, ev, color=SERIES[1], linewidth=2, marker="o", markersize=5, label="validation loss")
        i = min(range(len(ev)), key=ev.__getitem__)
        ax.scatter([es[i]], [ev[i]], s=90, facecolors="none", edgecolors=TEXT_PRIMARY, linewidths=1.5, zorder=5)
        ax.annotate(f"best {ev[i]:.3f} @ {es[i]}", (es[i], ev[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color=TEXT_PRIMARY)
    _style_axes(ax, "Loss — train vs validation", "cross-entropy")
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_SECONDARY)
    return True


def _plot_single(ax, rows, key: str, title: str, ylabel: str, color: str) -> bool:
    s, v = _series(rows, key)
    if not s:
        return False
    ax.plot(s, v, color=color, linewidth=2)
    _style_axes(ax, title, ylabel)
    return True


PANELS = [
    ("train/learning_rate", "Learning rate", "lr"),
    ("train/grad_norm", "Gradient norm", "L2 norm"),
    ("eval/perplexity", "Validation perplexity", "perplexity"),
    ("perf/tokens_per_sec", "Throughput", "tokens / sec"),
    ("perf/memory_gb", "Peak memory", "GB"),
]


def make_plots(run_dir: Path) -> dict[str, Path]:
    """Write ``plots/*.png``; returns {name: path}. Safe to call any time during training."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots")
        return {}

    rows = read_metrics(run_dir)
    if not rows:
        return {}
    out_dir = Path(run_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    if _plot_loss(ax, rows):
        fig.tight_layout()
        fig.savefig(out_dir / "loss.png", dpi=130)
        written["loss"] = out_dir / "loss.png"
    plt.close(fig)

    for key, title, ylabel in PANELS:
        fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=SURFACE)
        if _plot_single(ax, rows, key, title, ylabel, SERIES[0]):
            fig.tight_layout()
            name = key.split("/")[-1]
            fig.savefig(out_dir / f"{name}.png", dpi=120)
            written[name] = out_dir / f"{name}.png"
        plt.close(fig)
    return written
