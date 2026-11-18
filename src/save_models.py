#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/save_models.py
==================================================================
Persist the FINAL, reusable classifiers so a live demo / interface can
load them and score a brand-new peptide WITHOUT retraining.

For each dataset we save the canonical seed-0, full-320 version of every
model (the same recipe used in models.py / the benchmark):

    logreg, svm, rf, xgb  -> models_saved/<dataset>__<model>.joblib
        bundle = {"kind": model, "scaler": MinMaxScaler (fit on TRAIN core),
                  "model": fitted estimator, "dataset": name}

The CNN is already saved by the masking stage
(results/masking_models/<dataset>_cnn.pt + _scaler.pkl); the inference
module loads it from there, so we do NOT duplicate it here - we just make
sure it exists by calling build_predictor().

These saved objects take a MIN-MAX-SCALED 320-D ESM-2 embedding as input,
exactly matching how they were trained. src/inference.py wraps the
embedding + scaling + predict so callers only ever pass raw sequences.

Run:  python -m src.save_models            # all datasets, all models
      python -m src.save_models acp         # one dataset
"""

import os
import sys

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner, set_all_seeds
from src.embed import load_cache
from src.split import make_train_val

from sklearn.preprocessing import MinMaxScaler

SAVED_DIR = os.path.join(CFG.ROOT, "models_saved")
os.makedirs(SAVED_DIR, exist_ok=True)

CANON_SEED = 0                       # the single demo model per dataset
CLASSICAL = ["logreg", "svm", "rf", "xgb"]


def _fit_classical(kind, Xtr, ytr, seed=CANON_SEED):
    """Fit one classical model with the SAME CV grid as models.train_classical,
    but RETURN the fitted best estimator (models.py only returns predictions)."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    if kind == "logreg":
        est = LogisticRegression(max_iter=2000, class_weight="balanced")
        grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    elif kind == "svm":
        est = SVC(kernel="rbf", probability=True, class_weight="balanced",
                  random_state=seed)
        grid = {"C": [0.5, 1.0, 10.0], "gamma": ["scale", 0.01]}
    elif kind == "rf":
        est = RandomForestClassifier(class_weight="balanced", random_state=seed,
                                     n_jobs=-1)
        grid = {"n_estimators": [200, 400], "max_depth": [None, 10]}
    elif kind == "xgb":
        from xgboost import XGBClassifier
        pos = max(1, int((ytr == 1).sum()))
        neg = max(1, int((ytr == 0).sum()))
        est = XGBClassifier(eval_metric="logloss", random_state=seed,
                            n_jobs=-1, scale_pos_weight=neg / pos,
                            tree_method="hist")
        grid = {"n_estimators": [200, 400], "max_depth": [3, 6],
                "learning_rate": [0.1]}
    else:
        raise ValueError(kind)

    gs = GridSearchCV(est, grid, scoring="matthews_corrcoef", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_, gs.best_params_


def save_dataset(name, force=False):
    banner("saving demo models - {}".format(name))
    cache = load_cache(name)
    X, y, split = cache["X"], cache["y"], cache["split"]
    tr_all = np.where(split == "train")[0]
    core_rel, _ = make_train_val(y[tr_all], CANON_SEED)
    tr = tr_all[core_rel]                         # TRAIN core, seed 0

    scaler = MinMaxScaler().fit(X[tr])            # fit on TRAIN core only
    Xtr = scaler.transform(X[tr])
    ytr = y[tr]

    for kind in CLASSICAL:
        out = os.path.join(SAVED_DIR, "{}__{}.joblib".format(name, kind))
        if os.path.exists(out) and not force:
            print("  [skip] {} already saved".format(os.path.basename(out)))
            continue
        set_all_seeds(CANON_SEED)
        est, params = _fit_classical(kind, Xtr, ytr)
        joblib.dump({"kind": kind, "scaler": scaler, "model": est,
                     "dataset": name, "seed": CANON_SEED}, out)
        print("  saved {:<28} best_params={}".format(os.path.basename(out), params))

    # make sure the canonical CNN exists (saved by the masking stage)
    from src.masking import build_predictor
    build_predictor(name)                         # trains+saves if missing
    print("  CNN ready: results/masking_models/{}_cnn.pt".format(name))


def main():
    datasets = [a for a in sys.argv[1:] if a in CFG.DATASETS] or CFG.DATASETS
    for name in datasets:
        save_dataset(name)
    print("\n  all demo models saved under models_saved/ (+ masking_models/ for CNN).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
