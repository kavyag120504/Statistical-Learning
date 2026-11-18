#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/pca_ablation.py
==================================================================
Preprocessing + PCA ablation for the UniDL4BioPep project.

Two jobs:

  1. build_representations(Xtr, Xval, Xte)  -- the function the model
     scripts call. It:
       * MIN-MAX normalises each of the 320 features, fitting min/max on
         TRAIN ONLY, then applies the same transform to val and test;
       * builds four representations of the data:
            full  = all 320 features (just min-max scaled)
            pca90 = PCA keeping >=90 % variance
            pca95 = PCA keeping >=95 % variance
            pca99 = PCA keeping >=99 % variance
         Every PCA is FIT ON TRAIN ONLY (after scaling) and applied to
         val/test. Returns a dict repr_name -> (Xtr, Xval, Xte, n_comp).

  2. main() -- reports, per dataset, how many components each variance
     target needs (fit on the whole TRAIN split) and writes
     results/pca_components.csv. This answers project question 3's
     "record the number of components kept".

Nothing here is ever fit on validation or test data.

Run:  python -m src.pca_ablation
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner
from src.embed import load_cache

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def build_representations(Xtr, Xval, Xte, variance_targets=CFG.PCA_VARIANCE):
    """Return {repr_name: (Xtr, Xval, Xte, n_components)}.

    Fits MinMax + PCA on TRAIN only. 'full' keeps all 320 scaled features.
    """
    scaler = MinMaxScaler()
    Str = scaler.fit_transform(Xtr)           # fit on TRAIN
    Sval = scaler.transform(Xval)
    Ste = scaler.transform(Xte)

    reps = {"full": (Str, Sval, Ste, Str.shape[1])}

    for v in variance_targets:
        pca = PCA(n_components=v, svd_solver="full", random_state=0)
        Ptr = pca.fit_transform(Str)          # fit on TRAIN only
        Pval = pca.transform(Sval)
        Pte = pca.transform(Ste)
        tag = "pca{}".format(int(round(v * 100)))
        reps[tag] = (Ptr, Pval, Pte, pca.n_components_)
    return reps


def variance_report(name):
    """Fit PCA on the full TRAIN split and report components per target."""
    d = load_cache(name)
    tr = d["split"] == "train"
    Xtr = d["X"][tr]
    scaler = MinMaxScaler()
    Str = scaler.fit_transform(Xtr)

    # one full PCA, then read off components needed for each target
    pca = PCA(svd_solver="full", random_state=0).fit(Str)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    row = {"dataset": name, "n_train": int(tr.sum()), "total_dims": Xtr.shape[1]}
    for v in CFG.PCA_VARIANCE:
        n_comp = int(np.searchsorted(cumvar, v) + 1)
        row["pca{}_comp".format(int(round(v * 100)))] = n_comp
    return row, cumvar


def main():
    banner("pca_ablation.py - min-max + PCA variance report (TRAIN only)")
    rows = []
    curves = {}
    for name in CFG.DATASETS:
        print("\n>>> {}".format(name))
        row, cumvar = variance_report(name)
        curves[name] = cumvar
        rows.append(row)
        msg = "  components for variance targets: " + ", ".join(
            "{}%={}".format(int(round(v * 100)), row["pca{}_comp".format(int(round(v * 100)))])
            for v in CFG.PCA_VARIANCE)
        print(msg + "  (of {} dims)".format(row["total_dims"]))

    summ = pd.DataFrame(rows)
    out = os.path.join(CFG.RESULTS, "pca_components.csv")
    summ.to_csv(out, index=False)
    # also save the cumulative-variance curves for the PCA figure later
    np.savez(os.path.join(CFG.RESULTS, "pca_curves.npz"),
             **{k: v for k, v in curves.items()})

    banner("PCA COMPONENTS NEEDED (fit on TRAIN)")
    print(summ.to_string(index=False))
    print("\n  wrote {}".format(os.path.relpath(out, CFG.ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
