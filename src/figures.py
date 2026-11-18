#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/figures.py
==================================================================
Make every figure the project asks for, in a plain black-and-white
style with BLUE borders and large readable fonts, saved as BOTH PDF and
PNG in figures/.

Figures produced (when the needed results exist):
  1. model_comparison_<dataset>   - MCC per model (full repr), std bars
  2. pca_mcc_<dataset>            - MCC vs representation / #components
  3. peptide_importance_<dataset> - residue-importance heat strips for
                                     5 example active peptides
  4. deletion_test_<dataset>      - mean P(active) drop: top vs random vs bottom
  5. position_importance          - importance vs relative position (all datasets)

Run:  python -m src.figures
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner
from src.masking import MASK_OUT_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f4fd8"
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 110,
    "axes.edgecolor": BLUE, "axes.linewidth": 1.8,
})


def _style(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor(BLUE)
        sp.set_linewidth(1.8)
    ax.tick_params(colors="black")


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(CFG.FIGURES, "{}.{}".format(name, ext)),
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/{}.pdf / .png".format(name))


# ------------------------------------------------------------------ #
# 1. model comparison (MCC, full repr)
# ------------------------------------------------------------------ #
def fig_model_comparison(name, allm):
    sub = allm[(allm["dataset"] == name) & (allm["repr"] == "full")]
    if sub.empty:
        return
    sub = sub.sort_values("mcc_mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(sub["model"], sub["mcc_mean"], xerr=sub["mcc_std"],
            color="0.8", edgecolor="black", capsize=4)
    for y, v in zip(range(len(sub)), sub["mcc_mean"]):
        ax.text(v + 0.005, y, "{:.3f}".format(v), va="center", fontsize=11)
    ax.set_xlabel("MCC (test, mean +/- std over 5 seeds)")
    ax.set_title("{}: model comparison (full 320-D)".format(name))
    _style(ax)
    _save(fig, "model_comparison_{}".format(name))


# ------------------------------------------------------------------ #
# 2. PCA variance vs MCC
# ------------------------------------------------------------------ #
def fig_pca_mcc(name, allm):
    sub = allm[allm["dataset"] == name]
    if sub.empty:
        return
    order = ["full", "pca99", "pca95", "pca90"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"cnn": "o", "xgb": "s", "svm": "^", "rf": "D", "logreg": "v"}
    for model in sub["model"].unique():
        m = sub[sub["model"] == model].set_index("repr").reindex(order)
        ax.plot(order, m["mcc_mean"], marker=markers.get(model, "o"),
                color="black", label=model, linewidth=1.4, markersize=7,
                markerfacecolor="white")
        ax.errorbar(order, m["mcc_mean"], yerr=m["mcc_std"], fmt="none",
                    ecolor="0.5", capsize=3)
    ax.set_ylabel("MCC (test)")
    ax.set_xlabel("representation (full = 320-D, then PCA at 99/95/90% var)")
    ax.set_title("{}: PCA compression vs MCC".format(name))
    ax.legend(fontsize=10, ncol=2)
    _style(ax)
    _save(fig, "pca_mcc_{}".format(name))


# ------------------------------------------------------------------ #
# 3. per-peptide residue-importance heat strips
# ------------------------------------------------------------------ #
def fig_peptide_importance(name, n_examples=5, style="X"):
    path = os.path.join(MASK_OUT_DIR, "{}__{}__active.pkl".format(name, style))
    if not os.path.exists(path):
        return
    with open(path, "rb") as fh:
        recs = pickle.load(fh)
    if not recs:
        return
    # most confident active peptides as examples
    recs = sorted(recs, key=lambda r: r["p_full"], reverse=True)[:n_examples]
    fig, axes = plt.subplots(len(recs), 1, figsize=(10, 1.5 * len(recs)))
    if len(recs) == 1:
        axes = [axes]
    vmax = max(np.abs(r["residue_importance"]).max() for r in recs) or 1.0
    for ax, r in zip(axes, recs):
        imp = r["residue_importance"].reshape(1, -1)
        ax.imshow(imp, aspect="auto", cmap="Greys", vmin=0, vmax=vmax)
        ax.set_yticks([])
        seq = r["seq"]
        step = max(1, len(seq) // 40)
        ax.set_xticks(range(0, len(seq), step))
        ax.set_xticklabels(list(seq[::step]), fontsize=8)
        ax.set_ylabel("p={:.2f}".format(r["p_full"]), fontsize=10, rotation=0,
                      labelpad=28, va="center")
        _style(ax)
    axes[0].set_title("{}: residue importance (darker = model-dependent), style={}".format(name, style))
    _save(fig, "peptide_importance_{}".format(name))


# ------------------------------------------------------------------ #
# 4. deletion test
# ------------------------------------------------------------------ #
def fig_deletion(name):
    path = os.path.join(CFG.RESULTS, "deletion_raw_{}.csv".format(name))
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    means = [df["drop_top"].mean(), df["drop_rand"].mean(), df["drop_bottom"].mean()]
    stds = [df["drop_top"].std(), df["drop_rand"].std(), df["drop_bottom"].std()]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["top-m", "random-m", "bottom-m"]
    ax.bar(labels, means, yerr=stds, color=["0.4", "0.7", "0.9"],
           edgecolor="black", capsize=5)
    ax.set_ylabel("drop in P(active) when masked")
    ax.set_title("{}: deletion test (n={})".format(name, len(df)))
    _style(ax)
    _save(fig, "deletion_test_{}".format(name))


# ------------------------------------------------------------------ #
# 5. position importance (all datasets)
# ------------------------------------------------------------------ #
def fig_position():
    path = os.path.join(CFG.RESULTS, "position_profile.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = ["-", "--", "-.", ":"]
    for i, name in enumerate(df["dataset"].unique()):
        s = df[df["dataset"] == name]
        ax.plot(s["rel_position"], s["mean_importance"], styles[i % 4],
                color="black", marker="o", markerfacecolor="white",
                label=name, linewidth=1.5)
    ax.set_xlabel("relative position (0 = N-terminus, 1 = C-terminus)")
    ax.set_ylabel("mean residue importance")
    ax.set_title("Importance by position (active peptides)")
    ax.legend(fontsize=10)
    _style(ax)
    _save(fig, "position_importance")


def main():
    banner("figures.py - PDF + PNG, black/white, blue borders")
    allm_path = os.path.join(CFG.RESULTS, "all_metrics.csv")
    allm = pd.read_csv(allm_path) if os.path.exists(allm_path) else pd.DataFrame()
    for name in CFG.DATASETS:
        if not allm.empty:
            fig_model_comparison(name, allm)
            fig_pca_mcc(name, allm)
        fig_peptide_importance(name)
        fig_deletion(name)
    fig_position()
    print("\n  figures done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
