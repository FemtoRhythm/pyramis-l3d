"""Draw two figures for paper.pdf using the verified three-stage numbers.

Figures:
  1) fig_three_stage.pdf  -- grouped bars for utilization / PPL / norm across S1, S2, S3
  2) fig_norm_by_layer.pdf -- per-block codebook norm, demonstrating depth-monotone growth

Run:
  python3 /workspace/paper/make_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared publication-ready desaturated palette (used in BOTH figures for visual consistency)
COLOR_GREEN  = "#2e7d32"  # darker green  -> codebook norm series
COLOR_BLUE   = "#1b4965"  # muted dark blue
COLOR_ORANGE = "#c8731e"  # warm orange

# Shared text / grid styling
TXT_COLOR     = "#333333"
GRID_COLOR    = "#cccccc"
GRID_ALPHA    = 0.55
VALUE_FS      = 7
LABEL_FS      = 10
TITLE_FS      = 10
LEGEND_FS     = 8

# Verified three-stage data (from user's checkpoint_stage3 eval)
STAGES = ["S1 (ctx only)", "S2 (+EMA+revival+entropy)", "S3 (+detach removed)"]
UTIL = [0.1, (66 + 77) / 2, (60 + 70) / 2]  # % codebook utilization, averaged range midpoints
PPL  = [1.03, 1.091, 1.018]
NORM_RANGE_LO = [1e-3, 1e-3, 0.0306]
NORM_RANGE_HI = [1e-3, 1e-3, 2.8882]
NORM_MID = [(lo + hi) / 2 for lo, hi in zip(NORM_RANGE_LO, NORM_RANGE_HI)]

# Per-block norms (S3 final)
BLOCKS = ["Block 0", "Block 1", "Block 2", "Block 3"]
CB_NORM = [0.0306, 0.4135, 1.4000, 2.8882]
F_PROJ_NORM = [0.8395, 0.9830, 1.1312, 1.0096]
K_UP_NORM = [0.8949, 0.8944, 1.8423, 2.4784]


def _apply_axis_style(ax, ylabel, title, ylim=None, log=False):
    """Unified axis styling: white background, light gray major grid only, clean padding."""
    ax.tick_params(colors=TXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.6)
    ax.set_facecolor("white")
    ax.set_ylabel(ylabel, labelpad=6, fontsize=LABEL_FS, color=TXT_COLOR)
    ax.set_title(title, fontsize=TITLE_FS, pad=12, color=TXT_COLOR)
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Light gray major grid only, minor grid removed entirely
    ax.grid(axis="y", which="major", color=GRID_COLOR,
            linestyle="-", linewidth=0.6, alpha=GRID_ALPHA)
    ax.minorticks_off()
    ax.set_axisbelow(True)


def _add_value_labels(ax, bars, vals, fmt="{:.3f}", mult=1.18, color=TXT_COLOR):
    """Place a numeric value label above every bar with a consistent vertical offset."""
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v * mult, fmt.format(v),
                ha="center", va="bottom",
                fontsize=VALUE_FS, color=color)


def fig_three_stage():
    # Three fully independent subplots, one per metric -- zero dual axes.
    # Each panel uses identical bar width, tick sizes, and axis styling.
    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(6.5, 6.4),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.42},
    )
    x = np.arange(len(STAGES))
    width = 0.42  # single-series bars, can be wider than grouped bars

    # ---- Panel 1: Codebook utilization (%) ----
    bars_u = axes[0].bar(x, UTIL, width, color=COLOR_BLUE)
    _apply_axis_style(axes[0],
                      ylabel="Utilization (%)",
                      title="(a) Codebook Utilization",
                      ylim=(0, 90))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(STAGES, fontsize=9)
    _add_value_labels(axes[0], bars_u, UTIL, fmt="{:.1f}", mult=1.04)

    # ---- Panel 2: Held-out perplexity (PPL) ----
    bars_p = axes[1].bar(x, PPL, width, color=COLOR_ORANGE)
    _apply_axis_style(axes[1],
                      ylabel="Perplexity (PPL)",
                      title="(b) Held-Out Perplexity",
                      ylim=(0.98, 1.12))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(STAGES, fontsize=9)
    _add_value_labels(axes[1], bars_p, PPL, fmt="{:.3f}", mult=1.03)

    # ---- Panel 3: Codebook entry mean ℓ₂ norm (log) ----
    norm_plot = [max(n, 1e-4) for n in NORM_MID]
    bars_n = axes[2].bar(x, norm_plot, width, color=COLOR_GREEN)
    _apply_axis_style(axes[2],
                      ylabel="Mean entry $\\ell_2$ norm",
                      title="(c) Codebook Entry Magnitude (log scale)",
                      ylim=(5e-5, 6),
                      log=True)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(STAGES, fontsize=9)
    axes[2].set_xlabel("Ablation stage", labelpad=8, fontsize=LABEL_FS, color=TXT_COLOR)

    # Clean power-of-ten ticks on log axis, no minor grid
    axes[2].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    axes[2].minorticks_off()

    # Value labels on log scale: use clean decimal when magnitude >= 0.01,
    # fall back to 10^n only for truly tiny values -- avoids dense "1e-04" text.
    for b, v in zip(bars_n, norm_plot):
        if v >= 0.01:
            label = f"{v:.3f}"
        elif v >= 0.001:
            label = f"{v:.3f}"
        else:
            label = f"{v:.2e}"
        axes[2].text(b.get_x() + b.get_width() / 2,
                     v * 1.25, label,
                     ha="center", va="bottom", fontsize=VALUE_FS, color=TXT_COLOR)

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.14, right=0.92, top=0.97, bottom=0.08)
    out = os.path.join(OUT_DIR, "fig_three_stage.pdf")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def fig_norm_by_layer():
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(BLOCKS))
    width = 0.13  # ~28% narrower than 0.18
    bw = 1.6 * width  # bar-center offset -> intra-group spacing

    bars_cb = ax.bar(x - bw, CB_NORM, width, color=COLOR_GREEN,
                     label="Codebook mean $\\ell_2$ norm")
    bars_fp = ax.bar(x,     F_PROJ_NORM, width, color=COLOR_BLUE,
                     label="$f_{\\text{proj}}$ weight Frobenius norm")
    bars_ku = ax.bar(x + bw, K_UP_NORM, width, color=COLOR_ORANGE,
                     label="$k_{\\text{up}}$ weight Frobenius norm")

    ax.set_xticks(x)
    ax.set_xticklabels(BLOCKS, fontsize=9)
    ax.set_xlabel("Transformer block", labelpad=8, fontsize=LABEL_FS, color=TXT_COLOR)
    _apply_axis_style(ax,
                      ylabel="Norm magnitude",
                      title="S3 Codebook Norm Monotone Growth with Depth",
                      ylim=(1e-2, 5.5),
                      log=True)

    # Numeric value labels above EVERY bar, consistent vertical offset
    _add_value_labels(ax, bars_cb, CB_NORM,      fmt="{:.3f}", mult=1.18)
    _add_value_labels(ax, bars_fp, F_PROJ_NORM,  fmt="{:.3f}", mult=1.18)
    _add_value_labels(ax, bars_ku, K_UP_NORM,    fmt="{:.3f}", mult=1.18)

    # Compact horizontal legend below the x-axis title, outside plot area
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=LEGEND_FS, frameon=False,
              handlelength=1.4, columnspacing=1.5,
              handletextpad=0.5, borderpad=0.2)

    # Clean white background and consistent padding
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=1.5)
    out = os.path.join(OUT_DIR, "fig_norm_by_layer.pdf")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_three_stage()
    fig_norm_by_layer()
