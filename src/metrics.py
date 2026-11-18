#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/metrics.py
==================================================================
Turn the cached test predictions (results/predictions/*.npz) into the
per-dataset summary tables the project asks for.

For every (dataset x model x representation) cell we:
  * load the 5 seeds' P(active) predictions on TEST,
  * compute 8 binary metrics at threshold 0.5:
       Accuracy, Balanced Accuracy, Sensitivity (recall of active),
       Specificity, F1, MCC (main metric), ROC-AUC, PR-AUC,
  * report mean +/- std across the 5 seeds,
  * add a 95 % bootstrap confidence interval for MCC on the TEST set
    (resampling test peptides; uses the seed-averaged probabilities).

Outputs one CSV per dataset (results/summary_<dataset>.csv) plus a
combined results/all_metrics.csv.

Run:  python -m src.metrics
"""

import os
import sys
import glob
import re

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner
from src.models import PRED_DIR, ALL_MODELS

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, confusion_matrix,
)

N_BOOT = 2000


def _metrics_at(y_true, y_prob, thr=0.5):
    """All 8 metrics for one prediction vector."""
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0          # recall of active
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    # AUCs need both classes present and a probability
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def _bootstrap_mcc(y_true, y_prob, n=N_BOOT, seed=0):
    """95 % bootstrap CI for MCC by resampling test peptides."""
    rng = np.random.RandomState(seed)
    m = len(y_true)
    vals = []
    y_pred = (y_prob >= 0.5).astype(int)
    for _ in range(n):
        idx = rng.randint(0, m, m)
        # need both classes in the resample for MCC to be defined
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(matthews_corrcoef(y_true[idx], y_pred[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def collect(name):
    """Build the summary table for one dataset across models x reprs."""
    rows = []
    for model in ALL_MODELS:
        for repr_name in CFG.REPRS:
            files = sorted(glob.glob(os.path.join(
                PRED_DIR, "{}__{}__{}__seed*.npz".format(name, model, repr_name))))
            if not files:
                continue
            per_seed = []
            probs, ytrue = [], None
            for f in files:
                d = np.load(f)
                yt, yp = d["y_true"], d["y_prob"]
                per_seed.append(_metrics_at(yt, yp))
                probs.append(yp)
                ytrue = yt
            dfm = pd.DataFrame(per_seed)
            mean, std = dfm.mean(), dfm.std(ddof=0)
            # seed-ensemble: average the 5 seeds' probabilities, then score once.
            # The bootstrap CI (test-set resampling) is anchored to THIS ensemble
            # point MCC, so the CI brackets mcc_ensemble, not the per-seed mcc_mean
            # (ensembling usually lifts MCC above the mean of individual seeds).
            avg_prob = np.mean(np.vstack(probs), axis=0)
            mcc_ensemble = matthews_corrcoef(ytrue, (avg_prob >= 0.5).astype(int))
            lo, hi = _bootstrap_mcc(ytrue, avg_prob)
            row = {"dataset": name, "model": model, "repr": repr_name,
                   "n_seeds": len(files),
                   "n_comp": int(np.load(files[0])["n_comp"])}
            for k in dfm.columns:
                row[k + "_mean"] = round(float(mean[k]), 4)
                row[k + "_std"] = round(float(std[k]), 4)
            row["mcc_ensemble"] = round(float(mcc_ensemble), 4)
            row["mcc_ci_lo"] = round(lo, 4)
            row["mcc_ci_hi"] = round(hi, 4)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    banner("metrics.py - per-dataset summary tables (mean+/-std, bootstrap MCC CI)")
    datasets = sys.argv[1:] or CFG.DATASETS
    all_rows = []
    for name in datasets:
        df = collect(name)
        if df.empty:
            print("  [skip] no predictions for {}".format(name))
            continue
        df = df.sort_values(["repr", "mcc_mean"], ascending=[True, False])
        out = os.path.join(CFG.RESULTS, "summary_{}.csv".format(name))
        df.to_csv(out, index=False)
        all_rows.append(df)

        print("\n>>> {}  (MCC mean +/- std, with 95% bootstrap CI)".format(name))
        show = df[["model", "repr", "n_comp", "mcc_mean", "mcc_std",
                   "mcc_ci_lo", "mcc_ci_hi", "roc_auc_mean", "f1_mean"]]
        print(show.to_string(index=False))
        print("  wrote {}".format(os.path.relpath(out, CFG.ROOT)))

    if all_rows:
        comb = pd.concat(all_rows, ignore_index=True)
        comb.to_csv(os.path.join(CFG.RESULTS, "all_metrics.csv"), index=False)
        print("\n  wrote results/all_metrics.csv ({} rows)".format(len(comb)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
