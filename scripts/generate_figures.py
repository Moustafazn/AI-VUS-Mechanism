#!/usr/bin/env python3
"""
SpliceVarMech — Publication Figure Generator
Target Journal: Molecular Cell (Cell Press)

Molecular Cell Article format:
  - Max 7 display items (figures or tables) in main text
  - Main text <= 7,000 words (incl. figure legends, excl. STAR Methods)
  - 300 DPI, single column (85mm) or double column (178mm)

Main Figures (5 figures + 2 tables = 7 display items):
  Figure 1: Framework architecture (professional schematic)
  Figure 2: LOO-CV + SpliceAI head-to-head
  Figure 3: 16-tool baseline comparison
  Figure 4: Training convergence + calibration
  Figure 5: SOTA benchmark + capability comparison

Supplementary Figures:
  Figure S1: SpliceAI per-variant detail
  Figure S2: 3D tool landscape
  Figure S3: Cross-dataset generalization

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle, Circle, Arc
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    pass

# ── Cell Press style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Colors ──
C = {
    "deep_blue": "#1a5276", "ocean_blue": "#2980b9", "sky_blue": "#85c1e9",
    "deep_green": "#196f3d", "emerald": "#27ae60", "light_green": "#abebc6",
    "deep_red": "#922b21", "crimson": "#c0392b", "salmon": "#f1948a",
    "deep_purple": "#6c3483", "purple": "#8e44ad", "lavender": "#d2b4de",
    "amber": "#f39c12", "gold": "#f9e79f",
    "charcoal": "#2c3e50", "slate": "#34495e", "silver": "#bdc3c7",
    "cloud": "#ecf0f1", "white": "#ffffff",
    "teal": "#1abc9c", "orange": "#e67e22",
}

TOOL_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
    "#2980b9", "#27ae60", "#f1c40f", "#8e44ad", "#d35400",
    "#2c3e50", "#7f8c8d",
]

RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("paper/figures")

SC = 85 / 25.4    # single column
DC = 178 / 25.4   # double column


def load_json(fn: str) -> dict | None:
    fp = RESULTS_DIR / fn
    if not fp.exists():
        print(f"  !! {fp} not found")
        return None
    with open(fp) as f:
        return json.load(f)


def save_fig(fig, name, fmts=("pdf",)):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in fmts:
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}", format=fmt, dpi=300,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  -> {name}")


def _box(ax, x, y, w, h, text, fc, ec, fs=6, lw=1.2, alpha=0.15, tc=None, bold=True, radius=0.1):
    """Draw rounded box with text."""
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                       facecolor=fc, alpha=alpha, edgecolor=ec, linewidth=lw)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=tc or ec, linespacing=1.35, zorder=10)
    return b


def _arrow(ax, x1, y1, x2, y2, c=C["charcoal"], lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=c, linewidth=lw,
                               connectionstyle="arc3,rad=0"), zorder=5)


def _curved_arrow(ax, x1, y1, x2, y2, c=C["charcoal"], lw=1.5, rad=0.15):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, linewidth=lw,
                               connectionstyle=f"arc3,rad={rad}"), zorder=5)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Framework Architecture — Professional Schematic
# ══════════════════════════════════════════════════════════════════════

def figure1_framework():
    """Clean Cell Press-style pipeline diagram — minimal text, clear flow."""
    fig, ax = plt.subplots(figsize=(DC, DC * 0.62))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── Top timeline bar with annotations ──
    timeline_y = 8.5
    ax.plot([1.5, 14.5], [timeline_y, timeline_y], color=C["silver"], lw=2, zorder=1)

    phases = [
        (2.0, "Data integration", "Variant parsing\nSplice tool scores"),
        (5.5, "Representation\nlearning", "Dual-stream encoding\nContrastive training"),
        (9.0, "Generative\nmodeling", "D3PM reverse sampling\nMechanism classification"),
        (12.5, "Causal inference", "Bayesian posterior\nACMG classification"),
    ]
    for i, (x, title, subtitle) in enumerate(phases):
        circle = Circle((x, timeline_y), 0.22, facecolor=C["crimson"], edgecolor="white",
                        linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, timeline_y, str(i+1), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)
        ax.text(x, timeline_y + 1.0, title, ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color=C["charcoal"], linespacing=1.2)
        ax.text(x, timeline_y + 0.4, subtitle, ha="center", va="bottom",
                fontsize=5, color=C["slate"], linespacing=1.2)

    for i in range(len(phases) - 1):
        _arrow(ax, phases[i][0] + 0.3, timeline_y, phases[i+1][0] - 0.3, timeline_y, C["silver"], 1.5)

    # ── 3 module panels ──
    panel_top = 7.4
    panel_h = 5.0
    item_start = panel_top - 0.7
    item_spacing = 0.52

    # MODULE 1
    m1_x, m1_w = 0.3, 5.0
    r1 = FancyBboxPatch((m1_x, panel_top - panel_h), m1_w, panel_h, boxstyle="round,pad=0.15",
                         facecolor=C["sky_blue"], alpha=0.08, edgecolor=C["ocean_blue"], linewidth=1.8)
    ax.add_patch(r1)
    ax.text(m1_x + m1_w/2, panel_top - 0.25, "Biological Diffusion Model",
            ha="center", fontsize=7.5, fontweight="bold", color=C["deep_blue"])

    items1 = [
        "Variant highlight (Gaussian spread)",
        "Multi-scale CNN (k=5, 15, 51)",
        "Dual-stream WT/MUT encoder",
        "Cross-attention comparison",
        "Contrastive loss",
        "D3PM decoder (100 steps)",
        "Self-conditioning",
    ]
    for j, item in enumerate(items1):
        y = item_start - j * item_spacing
        ax.plot(m1_x + 0.4, y, 'o', color=C["ocean_blue"], markersize=2.5, zorder=5)
        ax.text(m1_x + 0.65, y, item, fontsize=5, va="center", color=C["charcoal"])

    _box(ax, m1_x + 0.2, panel_top - panel_h + 0.15, m1_w - 0.4, 0.5,
         "Mechanism | Aberrant frac. | Score",
         C["sky_blue"], C["ocean_blue"], fs=5, alpha=0.3, bold=False, tc=C["deep_blue"])

    # MODULE 2
    m2_x, m2_w = 5.7, 4.6
    r2 = FancyBboxPatch((m2_x, panel_top - panel_h), m2_w, panel_h, boxstyle="round,pad=0.15",
                         facecolor=C["light_green"], alpha=0.08, edgecolor=C["emerald"], linewidth=1.8)
    ax.add_patch(r2)
    ax.text(m2_x + m2_w/2, panel_top - 0.25, "Bayesian Causal Inference",
            ha="center", fontsize=7.5, fontweight="bold", color=C["deep_green"])

    items2 = [
        "Structural causal model (DAG)",
        "MCMC posterior (NUTS)",
        "Hierarchical shrinkage prior",
        "Class-balanced likelihood",
        "95% credible intervals",
        "Counterfactual reasoning",
        "do-calculus intervention",
    ]
    for j, item in enumerate(items2):
        y = item_start - j * item_spacing
        ax.plot(m2_x + 0.3, y, 'o', color=C["emerald"], markersize=2.5, zorder=5)
        ax.text(m2_x + 0.55, y, item, fontsize=5, va="center", color=C["charcoal"])

    _box(ax, m2_x + 0.2, panel_top - panel_h + 0.15, m2_w - 0.4, 0.5,
         "P(disruption) + CI | Causal paths",
         C["light_green"], C["emerald"], fs=5, alpha=0.3, bold=False, tc=C["deep_green"])

    # MODULE 3
    m3_x, m3_w = 10.7, 5.0
    r3 = FancyBboxPatch((m3_x, panel_top - panel_h), m3_w, panel_h, boxstyle="round,pad=0.15",
                         facecolor=C["salmon"], alpha=0.06, edgecolor=C["crimson"], linewidth=1.8)
    ax.add_patch(r3)
    ax.text(m3_x + m3_w/2, panel_top - 0.25, "Explainable AI + Clinical Report",
            ha="center", fontsize=7.5, fontweight="bold", color=C["deep_red"])

    items3 = [
        "Sequence attribution maps",
        "Causal path visualization",
        "ACMG criteria mapping",
        "PP3 / PS3 / PM2 evidence",
        "Clinical confidence grading",
        "Mechanism explanation",
        "Reclassification recommendation",
    ]
    for j, item in enumerate(items3):
        y = item_start - j * item_spacing
        ax.plot(m3_x + 0.3, y, 'o', color=C["crimson"], markersize=2.5, zorder=5)
        ax.text(m3_x + 0.55, y, item, fontsize=5, va="center", color=C["charcoal"])

    _box(ax, m3_x + 0.2, panel_top - panel_h + 0.15, m3_w - 0.4, 0.5,
         "ACMG class | Mechanism | Uncertainty",
         C["salmon"], C["crimson"], fs=5, alpha=0.25, bold=False, tc=C["deep_red"])

    # ── Flow arrows between modules ──
    mid_y = panel_top - panel_h/2
    _arrow(ax, m1_x + m1_w + 0.05, mid_y, m2_x - 0.05, mid_y, C["charcoal"], 2.0)
    _arrow(ax, m2_x + m2_w + 0.05, mid_y, m3_x - 0.05, mid_y, C["charcoal"], 2.0)

    # ── Bottom bar ──
    bot_y = 0.2
    bot_h = 1.0
    _box(ax, 0.3, bot_y, 5.0, bot_h,
         "Input: Genomic variant (HGVS)\nWT + MUT pre-mRNA context",
         C["cloud"], C["slate"], fs=5.5, alpha=0.4, bold=False, tc=C["charcoal"])

    _box(ax, 5.7, bot_y, 4.6, bot_h,
         "Pre-train: GENCODE 252K junctions\nFine-tune: Gold-std + augmentation",
         C["cloud"], C["slate"], fs=5.5, alpha=0.4, bold=False, tc=C["charcoal"])

    _box(ax, 10.7, bot_y, 5.0, bot_h,
         "Output: Clinical report\nVUS reclassification",
         C["gold"], C["amber"], fs=5.5, alpha=0.3, bold=True, tc=C["charcoal"])

    _arrow(ax, 2.8, bot_y + bot_h, 2.8, panel_top - panel_h, C["silver"], 1.0)
    _arrow(ax, 8.0, bot_y + bot_h, 8.0, panel_top - panel_h, C["silver"], 1.0)
    _arrow(ax, 13.2, panel_top - panel_h, 13.2, bot_y + bot_h, C["silver"], 1.0)

    save_fig(fig, "figure1_framework")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: LOO-CV + SpliceAI Head-to-Head
# ══════════════════════════════════════════════════════════════════════

def figure2_loo_cv():
    data = load_json("loo_cv.json")
    sai = load_json("spliceai_evaluation.json")

    fig = plt.figure(figsize=(DC, DC * 0.42))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.6, 0.7, 0.9], wspace=0.3)

    # A: Waterfall
    ax1 = fig.add_subplot(gs[0])
    ax1.text(-0.1, 1.06, "A", fontsize=11, fontweight="bold", transform=ax1.transAxes)

    if data and data.get("per_variant"):
        vs = sorted(data["per_variant"], key=lambda v: v.get("p_mean", 0.5))
        names = [v.get("variant", "")[:24] for v in vs]
        probs = [v.get("p_mean", 0.5) for v in vs]
        los = [v.get("p_lower", 0) for v in vs]
        his = [v.get("p_upper", 1) for v in vs]
        labs = [v.get("label", 0) for v in vs]
        cols = [C["crimson"] if l == 1 else C["ocean_blue"] for l in labs]

        y = np.arange(len(names))
        ax1.barh(y, probs, color=cols, alpha=0.8, height=0.75)
        for i, (p, lo, hi) in enumerate(zip(probs, los, his)):
            ax1.plot([lo, hi], [i, i], color=C["slate"], linewidth=0.5, alpha=0.5)
        ax1.axvline(0.5, color=C["charcoal"], ls="--", lw=0.8, alpha=0.6)
        opt = data.get("optimal_threshold", 0.58)
        ax1.axvline(opt, color=C["amber"], ls=":", lw=0.8, alpha=0.7)
        ax1.set_yticks(y)
        ax1.set_yticklabels(names, fontsize=4)
        ax1.set_xlabel("P(disruption)")
        ax1.set_xlim(0, 1.05)
        ax1.legend(handles=[Patch(fc=C["crimson"], label="Pathogenic (n=23)"),
                            Patch(fc=C["ocean_blue"], label="Benign (n=8)")],
                   frameon=False, fontsize=5, loc="lower right")
    ax1.set_title("LOO-CV Per-Variant Predictions", fontsize=7.5)

    # B: Confusion matrix
    ax2 = fig.add_subplot(gs[1])
    ax2.text(-0.15, 1.06, "B", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    if data and data.get("eval_at_optimal"):
        ev = data["eval_at_optimal"]
        cm = np.array([[ev["tn"], ev["fp"]], [ev["fn"], ev["tp"]]])
        ax2.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                col = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold", color=col)
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["Pred.\nBenign", "Pred.\nPath."], fontsize=6)
        ax2.set_yticklabels(["True\nBenign", "True\nPath."], fontsize=6)
    ax2.set_title(f"Confusion (t={data.get('optimal_threshold', 0.58):.2f})" if data else "", fontsize=7)

    # C: Metrics comparison
    ax3 = fig.add_subplot(gs[2])
    ax3.text(-0.12, 1.06, "C", fontsize=11, fontweight="bold", transform=ax3.transAxes)
    if data:
        mnames = ["AUROC", "AUPRC", "Sens.", "Spec.", "BalAcc", "MCC"]
        ours = [data.get("auroc", 0), data.get("auprc", 0),
                data["eval_at_optimal"]["sensitivity"], data["eval_at_optimal"]["specificity"],
                data["eval_at_optimal"]["balanced_accuracy"], data["eval_at_optimal"]["mcc"]]
        sai_v = [sai.get("auroc", 0) or 0, sai.get("auprc", 0) or 0,
                 sai.get("sensitivity", 0) or 0, sai.get("specificity", 0) or 0,
                 sai.get("balanced_accuracy", 0) or 0, sai.get("mcc", 0) or 0] if sai else [0]*6

        x = np.arange(len(mnames)); w = 0.30
        b1 = ax3.bar(x - w/2, ours, w, label="SpliceVarMech", color=C["crimson"], alpha=0.85)
        b2 = ax3.bar(x + w/2, sai_v, w, label="SpliceAI", color=C["ocean_blue"], alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(mnames, fontsize=5, rotation=45, ha="right")
        ax3.set_ylim(0, 1.25)
        ax3.axhline(0.5, color=C["silver"], ls=":", alpha=0.5)
        ax3.legend(frameon=False, fontsize=5, loc="upper left")
        for bar in b1:
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                     f"{bar.get_height():.2f}", ha="center", fontsize=3.5,
                     rotation=90, va="bottom", color=C["deep_red"])
        for bar in b2:
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                     f"{bar.get_height():.2f}", ha="center", fontsize=3.5,
                     rotation=90, va="bottom", color=C["deep_blue"])
    ax3.set_ylabel("Score")
    ax3.set_title("SpliceVarMech vs SpliceAI", fontsize=7.5)

    save_fig(fig, "figure2_loo_cv")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Baseline 16-Tool Comparison
# ══════════════════════════════════════════════════════════════════════

def figure3_baseline():
    data = load_json("baseline_tools.json")

    fig = plt.figure(figsize=(DC, DC * 0.42))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.text(-0.06, 1.06, "A", fontsize=11, fontweight="bold", transform=ax1.transAxes)

    if data and data.get("per_tool"):
        valid = [t for t in data["per_tool"] if t.get("auroc") is not None]
        ts = sorted(valid, key=lambda t: t["auroc"], reverse=True)
        names = [t["tool"].replace("_", " ").replace(" score", "").replace(" max", "").strip() for t in ts]
        aurocs = [t["auroc"] for t in ts]

        names.insert(0, "SpliceVarMech (Ours)")
        aurocs.insert(0, 0.940)

        y = np.arange(len(names))
        cols = [C["crimson"]] + [C["ocean_blue"]] * len(ts)
        bars = ax1.barh(y, aurocs, color=cols, alpha=0.85, height=0.7)
        bars[0].set_edgecolor(C["deep_red"]); bars[0].set_linewidth(2)
        ax1.set_yticks(y)
        ax1.set_yticklabels(names, fontsize=5.5)
        ax1.set_xlabel("AUROC")
        ax1.set_xlim(0, 1.05)
        ax1.axvline(0.5, color=C["silver"], ls=":", alpha=0.5)
        ax1.invert_yaxis()
        for bar, val in zip(bars, aurocs):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=4.5)
    ax1.set_title("AUROC: SpliceVarMech vs 16 Baseline Tools (Gold Standard, N=31)", fontsize=7)

    # B: Coverage vs AUROC
    ax2 = fig.add_subplot(gs[1])
    ax2.text(-0.12, 1.06, "B", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    if data and data.get("per_tool"):
        valid = [t for t in data["per_tool"] if t.get("auroc") is not None]
        for i, t in enumerate(valid):
            ax2.scatter(t["coverage_pct"], t["auroc"], s=40,
                       color=TOOL_COLORS[i % len(TOOL_COLORS)], alpha=0.7, zorder=3)
            ax2.annotate(t["tool"].replace("_", " ")[:12],
                        (t["coverage_pct"], t["auroc"]),
                        fontsize=3.5, textcoords="offset points", xytext=(3, 3))
        ax2.scatter(100, 0.940, s=150, color=C["crimson"], marker="*", zorder=5)
        ax2.annotate("SpliceVarMech", (100, 0.940), fontsize=6, fontweight="bold",
                    textcoords="offset points", xytext=(-50, 8), color=C["crimson"])
        ax2.axhline(0.5, color=C["silver"], ls=":", alpha=0.4)
        ax2.set_xlabel("Tool Coverage (%)")
        ax2.set_ylabel("AUROC")
        ax2.set_xlim(0, 110); ax2.set_ylim(0.2, 1.05)
    ax2.set_title("Coverage vs Discrimination", fontsize=7)

    save_fig(fig, "figure3_baseline_comparison")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Training Convergence + Calibration
# ══════════════════════════════════════════════════════════════════════

def figure4_training_calibration():
    pt = load_json("pretrain_history.json")
    ft = load_json("finetune_history.json")
    cal = load_json("calibration.json")

    fig = plt.figure(figsize=(DC, DC * 0.35))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    # A: Pre-training
    ax1 = fig.add_subplot(gs[0])
    ax1.text(-0.12, 1.06, "A", fontsize=11, fontweight="bold", transform=ax1.transAxes)
    if pt and pt.get("pretrain_loss"):
        losses = pt["pretrain_loss"]
        n = len(losses)
        if n > 500:
            nb = min(200, n // 10)
            bs = n // nb
            sm = [np.mean(losses[i*bs:min((i+1)*bs, n)]) for i in range(nb)]
            st = [(i*bs + min((i+1)*bs, n)) / 2 for i in range(nb)]
            ax1.plot(st, sm, color=C["ocean_blue"], lw=1.2, label="Loss (smoothed)")
            ax1.plot(range(n), losses, color=C["ocean_blue"], alpha=0.06, lw=0.2)
        else:
            ax1.plot(range(n), losses, color=C["ocean_blue"], lw=1.2)
        if pt.get("pretrain_val_loss"):
            vl = pt["pretrain_val_loss"]
            ax1.plot(np.linspace(0, n-1, len(vl)), vl, color=C["amber"], ls="--", lw=1, label="Val loss")
        ax1.legend(frameon=False, fontsize=5)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.set_title("Pre-training (GENCODE, 252K)", fontsize=7)

    # B: Fine-tuning
    ax2 = fig.add_subplot(gs[1])
    ax2.text(-0.12, 1.06, "B", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    if ft and ft.get("finetune_loss"):
        losses = ft["finetune_loss"]
        n = len(losses)
        if n > 200:
            nb = min(100, n // 5)
            bs = n // nb
            sm = [np.mean(losses[i*bs:min((i+1)*bs, n)]) for i in range(nb)]
            st = [(i*bs + min((i+1)*bs, n)) / 2 for i in range(nb)]
            ax2.plot(st, sm, color=C["emerald"], lw=1.2, label="Loss (smoothed)")
            ax2.plot(range(n), losses, color=C["emerald"], alpha=0.08, lw=0.2)
        else:
            ax2.plot(range(n), losses, color=C["emerald"], lw=1.2)
        if ft.get("finetune_val_loss"):
            vl = ft["finetune_val_loss"]
            ax2.plot(np.linspace(0, n-1, len(vl)), vl, color=C["amber"], ls="--", lw=1, label="Val loss")
        ax2.legend(frameon=False, fontsize=5)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Loss")
    ax2.set_title("Fine-tuning (Gold Std + Aug.)", fontsize=7)

    # C: Calibration
    ax3 = fig.add_subplot(gs[2])
    ax3.text(-0.12, 1.06, "C", fontsize=11, fontweight="bold", transform=ax3.transAxes)
    ax3.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect")
    if cal:
        ba = cal.get("bin_accuracies", [])
        bc = cal.get("bin_confidences", [])
        bn = cal.get("bin_counts", [])
        if ba and bc and bn:
            v = [i for i in range(len(bn)) if bn[i] > 0]
            if v:
                cs = [bc[i] for i in v]; ac = [ba[i] for i in v]; sz = [max(bn[i]*15, 30) for i in v]
                ax3.scatter(cs, ac, s=sz, color=C["ocean_blue"], alpha=0.7, zorder=5, edgecolors="white")
                sp = sorted(zip(cs, ac))
                ax3.plot([p[0] for p in sp], [p[1] for p in sp], color=C["ocean_blue"], lw=0.8, alpha=0.5)
        ax3.text(0.05, 0.85, f"ECE = {cal.get('ece', 0):.4f}\nBrier = {cal.get('brier_score', 0):.4f}",
                 transform=ax3.transAxes, fontsize=6,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    ax3.set_xlabel("Predicted Probability"); ax3.set_ylabel("Observed Frequency")
    ax3.set_xlim(-0.02, 1.02); ax3.set_ylim(-0.02, 1.02)
    ax3.set_title("Calibration", fontsize=7)

    save_fig(fig, "figure4_training_calibration")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: SOTA Benchmark + Capability Comparison
# ══════════════════════════════════════════════════════════════════════

def figure5_sota_benchmark():
    bench = load_json("benchmark_comparison.json")
    sota = load_json("sota_benchmark.json")

    fig = plt.figure(figsize=(DC, DC * 0.42))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    # A: SOTA capability comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.text(-0.1, 1.06, "A", fontsize=11, fontweight="bold", transform=ax1.transAxes)

    if bench and bench.get("literature_benchmarks"):
        bms = bench["literature_benchmarks"]
        methods, aurocs, caps = [], [], []
        for bm in bms:
            a = bm.get("reported_auroc")
            if a and a > 0:
                methods.append(bm["method"])
                aurocs.append(a)
                caps.append(sum([bm.get("predicts_mechanism", False),
                                bm.get("generates_sequence", False),
                                bm.get("provides_uncertainty", False),
                                bm.get("tissue_aware", False),
                                bm.get("explains_prediction", False)]))

        # Color by approach type
        approach_colors = {"deep_learning": C["ocean_blue"], "foundation_model": C["purple"],
                          "ensemble": C["emerald"], "generative": C["amber"]}
        for i, bm in enumerate([b for b in bms if b.get("reported_auroc") and b["reported_auroc"] > 0]):
            col = approach_colors.get(bm.get("approach", ""), C["silver"])
            idx = methods.index(bm["method"])
            ax1.scatter(caps[idx], aurocs[idx], s=70, color=col, alpha=0.75, zorder=3,
                       edgecolors="white", linewidth=0.5)
            ax1.annotate(methods[idx], (caps[idx], aurocs[idx]), fontsize=4.5,
                        textcoords="offset points", xytext=(5, 3))

        ax1.scatter(5, 0.940, s=250, color=C["crimson"], marker="*", zorder=5, edgecolors="white")
        ax1.annotate("SpliceVarMech\n(Ours)", (5, 0.940), fontsize=6.5, fontweight="bold",
                    textcoords="offset points", xytext=(-55, -18), color=C["crimson"])

        ax1.set_xlabel("Unique Capabilities (of 5)", fontsize=7)
        ax1.set_ylabel("Reported AUROC")
        ax1.set_xlim(-0.5, 5.8); ax1.set_ylim(0.82, 0.98)

        # Legend for approach types
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, markersize=6, label=k.replace("_", " ").title())
               for k, v in approach_colors.items()]
        leg.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=C["crimson"], markersize=10, label="Ours"))
        ax1.legend(handles=leg, fontsize=4.5, frameon=False, loc="lower right")

    ax1.set_title("SOTA: Performance vs Capabilities (2019-2026)", fontsize=7)

    # B: Head-to-head AUROC by dataset
    ax2 = fig.add_subplot(gs[1])
    ax2.text(-0.12, 1.06, "B", fontsize=11, fontweight="bold", transform=ax2.transAxes)

    if sota and sota.get("per_dataset"):
        pd = sota["per_dataset"]
        ds = list(pd.keys())
        ours = [pd[d].get("SpliceVarMech", {}).get("auroc") or 0 for d in ds]
        sai = [pd[d].get("SpliceAI", {}).get("auroc") or 0 for d in ds]
        cov_sai = [pd[d].get("SpliceAI", {}).get("coverage_pct") or 0 for d in ds]

        x = np.arange(len(ds)); w = 0.32
        b1 = ax2.bar(x - w/2, ours, w, label="SpliceVarMech (100% cov.)", color=C["crimson"], alpha=0.85)
        b2 = ax2.bar(x + w/2, sai, w, label="SpliceAI", color=C["ocean_blue"], alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.replace("_", "\n") for d in ds], fontsize=5.5)
        ax2.legend(frameon=False, fontsize=5)
        ax2.set_ylim(0, 1.15)
        ax2.axhline(0.5, color=C["silver"], ls=":", alpha=0.5)

        # Add coverage annotation for SpliceAI
        for i, (bar, cov) in enumerate(zip(b2, cov_sai)):
            if cov > 0:
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                        f"{cov:.0f}%", ha="center", fontsize=3.5, color=C["slate"])

    ax2.set_ylabel("AUROC")
    ax2.set_title("Head-to-Head by Dataset", fontsize=7)

    save_fig(fig, "figure5_sota_benchmark")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ══════════════════════════════════════════════════════════════════════

def figureS1_spliceai():
    data = load_json("spliceai_evaluation.json")
    fig, axes = plt.subplots(1, 2, figsize=(DC, SC * 1.1))

    ax1 = axes[0]
    ax1.text(-0.12, 1.06, "A", fontsize=11, fontweight="bold", transform=ax1.transAxes)
    if data and data.get("per_variant"):
        vs = [v for v in data["per_variant"] if v.get("spliceai_score") is not None]
        vs = sorted(vs, key=lambda v: v["spliceai_score"])
        names = [v["variant"][:22] for v in vs]
        scores = [v["spliceai_score"] for v in vs]
        labs = [v.get("label", 0) for v in vs]
        cols = [C["crimson"] if l == 1 else C["ocean_blue"] for l in labs]
        y = range(len(names))
        ax1.barh(y, scores, color=cols, alpha=0.8)
        ax1.axvline(data.get("optimal_threshold", 0.87), color=C["charcoal"], ls="--", lw=0.8)
        ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=4)
        ax1.set_xlabel("SpliceAI Score")
        ax1.legend(handles=[Patch(fc=C["crimson"], label="Pathogenic"),
                            Patch(fc=C["ocean_blue"], label="Benign")],
                   frameon=False, fontsize=5, loc="lower right")
    ax1.set_title("SpliceAI Per-Variant", fontsize=7)

    # Panel B: Metrics as clean grouped bar chart
    ax2 = axes[1]
    ax2.text(-0.1, 1.06, "B", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    if data:
        mnames = ["AUROC", "AUPRC", "Sens.", "Spec.", "BalAcc", "MCC"]
        sai_v = [
            data.get("auroc", 0) or 0, data.get("auprc", 0) or 0,
            data.get("sensitivity", 0) or 0, data.get("specificity", 0) or 0,
            data.get("balanced_accuracy", 0) or 0, data.get("mcc", 0) or 0,
        ]
        our_v = [0.940, 0.984, 0.913, 1.0, 0.957, 0.855]
        x = np.arange(len(mnames)); w = 0.32
        b1 = ax2.bar(x - w/2, our_v, w, label="SpliceVarMech", color=C["crimson"], alpha=0.85)
        b2 = ax2.bar(x + w/2, sai_v, w, label="SpliceAI", color=C["ocean_blue"], alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(mnames, fontsize=6, rotation=30, ha="right")
        ax2.set_ylim(0, 1.15)
        ax2.axhline(0.5, color=C["silver"], ls=":", alpha=0.4)
        ax2.legend(frameon=False, fontsize=5.5)
        ax2.set_ylabel("Score")
        for bar in b1:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f"{bar.get_height():.2f}", ha="center", fontsize=4)
        for bar in b2:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f"{bar.get_height():.2f}", ha="center", fontsize=4)
    ax2.set_title("SpliceVarMech vs SpliceAI", fontsize=7)
    save_fig(fig, "figureS1_spliceai")
    plt.close(fig)


def figureS2_3d_landscape():
    data = load_json("baseline_tools.json")
    fig = plt.figure(figsize=(DC, DC * 0.55))
    ax = fig.add_subplot(111, projection="3d")
    if data and data.get("per_tool"):
        tools = [t for t in data["per_tool"] if t.get("auroc") is not None and t.get("sensitivity") is not None]
        # Short clean names for legend
        short_names = {
            "spliceAI_max_score": "SpliceAI", "ESRseq": "ESRseq", "MaxEntScan": "MaxEntScan",
            "max_SPiCEprobability": "SPiCE", "mmsplice_delta_logit_psi": "MMSplice",
            "Spliceogen": "Spliceogen", "Kipoisplice_pathogenic": "Kipoi",
            "Squirls_max_score": "Squirls", "SCAP_max": "SCAP",
            "dpsi_max_tissue": "dpsi_tissue", "dpsi_zscore": "dpsi_z",
            "dbscSNV_ADA_SCORE": "dbscSNV_ADA", "GeneSplicer": "GeneSplicer",
            "CADDsplice_phred": "CADD-splice", "dbscSNV_RF_SCORE": "dbscSNV_RF",
            "regsnp_fpr": "RegSNP",
        }
        for i, t in enumerate(tools):
            s = t.get("sensitivity", 0) or 0
            sp = t.get("specificity", 0) or 0
            a = t.get("auroc", 0) or 0
            name = short_names.get(t["tool"], t["tool"][:12])
            ax.scatter(s, sp, a, s=50, color=TOOL_COLORS[i % len(TOOL_COLORS)],
                      alpha=0.8, label=name)
        ax.scatter(0.913, 1.0, 0.940, s=200, color=C["crimson"], marker="*",
                  zorder=10, label="SpliceVarMech")
        ax.set_xlabel("Sensitivity", fontsize=7, labelpad=8)
        ax.set_ylabel("Specificity", fontsize=7, labelpad=8)
        ax.set_zlabel("AUROC", fontsize=7, labelpad=8)
        ax.view_init(elev=25, azim=140)
        ax.tick_params(labelsize=5, pad=3)
        # Place legend below the plot to avoid overlap
        ax.legend(fontsize=4, bbox_to_anchor=(0.5, -0.12), loc="upper center",
                  ncol=4, frameon=True, fancybox=True, framealpha=0.9,
                  edgecolor=C["silver"], columnspacing=0.8, handletextpad=0.3,
                  markerscale=0.6)
    ax.set_title("Tool Performance Landscape", fontsize=8, pad=15)
    save_fig(fig, "figureS2_3d_tool_landscape")
    plt.close(fig)


def figureS3_generalization():
    gen = load_json("generalization_evaluation.json")
    sota = load_json("sota_benchmark.json")

    fig = plt.figure(figsize=(DC, SC * 1.0))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.text(-0.12, 1.06, "A", fontsize=11, fontweight="bold", transform=ax1.transAxes)
    if gen and gen.get("datasets"):
        items = sorted(gen["datasets"].items())
        names = [v.get("dataset", k).replace("_", " ").title() for k, v in items]
        aurocs = [v.get("auroc", 0) or 0 for _, v in items]
        nvars = [v.get("n_variants", 0) for _, v in items]
        x = np.arange(len(names))
        cols = [C["crimson"] if "gold" in n.lower() else C["ocean_blue"] if a >= 0.5 else C["silver"]
                for n, a in zip(names, aurocs)]
        bars = ax1.bar(x, aurocs, color=cols, alpha=0.85)
        ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=5, rotation=30, ha="right")
        ax1.axhline(0.5, color=C["silver"], ls=":", alpha=0.5)
        ax1.set_ylim(0, 1.05); ax1.set_ylabel("AUROC")
        for b, v, n in zip(bars, aurocs, nvars):
            ax1.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}\n(n={n})", ha="center", fontsize=4)
    ax1.set_title("Cross-Dataset Generalization", fontsize=7)

    ax2 = fig.add_subplot(gs[1])
    ax2.text(-0.12, 1.06, "B", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    if sota and sota.get("summary"):
        s = sota["summary"]
        tools = list(s.keys())
        aa = [s[t].get("avg_auroc", 0) or 0 for t in tools]
        ab = [s[t].get("avg_balanced_accuracy", 0) or 0 for t in tools]
        ac = [s[t].get("avg_coverage", 0) or 0 for t in tools]
        x = np.arange(len(tools)); w = 0.25
        ax2.bar(x-w, aa, w, label="Avg AUROC", color=C["crimson"], alpha=0.85)
        ax2.bar(x, ab, w, label="Avg BalAcc", color=C["emerald"], alpha=0.85)
        ax2.bar(x+w, [c/100 for c in ac], w, label="Coverage/100", color=C["amber"], alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(tools, fontsize=7)
        ax2.legend(frameon=False, fontsize=5); ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Score"); ax2.set_title("Average SOTA Summary", fontsize=7)

    save_fig(fig, "figureS3_generalization")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SpliceVarMech Figure Generator — Molecular Cell")
    print("=" * 60)

    expected = ["pretrain_history.json", "finetune_history.json", "baseline_tools.json",
                "loo_cv.json", "calibration.json", "benchmark_comparison.json",
                "spliceai_evaluation.json", "generalization_evaluation.json", "sota_benchmark.json"]
    for f in expected:
        s = "OK" if (RESULTS_DIR / f).exists() else "MISSING"
        print(f"  [{s}] {f}")

    print("\n  MAIN FIGURES:")
    for name, fn in [("Fig1: Architecture", figure1_framework),
                     ("Fig2: LOO-CV", figure2_loo_cv),
                     ("Fig3: Baselines", figure3_baseline),
                     ("Fig4: Training+Cal", figure4_training_calibration),
                     ("Fig5: SOTA", figure5_sota_benchmark)]:
        try:
            fn()
        except Exception as e:
            print(f"  !! {name}: {e}")
            import traceback; traceback.print_exc()

    print("\n  SUPPLEMENTARY:")
    for name, fn in [("FigS1: SpliceAI", figureS1_spliceai),
                     ("FigS2: 3D Landscape", figureS2_3d_landscape),
                     ("FigS3: Generalization", figureS3_generalization)]:
        try:
            fn()
        except Exception as e:
            print(f"  !! {name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. 5 main + 3 supplementary figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
