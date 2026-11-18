#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/models.py
==================================================================
All classifiers for the UniDL4BioPep project, trained on IDENTICAL
splits and inputs so the comparison is fair.

Models
------
  * CNN  : a faithful replica of the UniDL4BioPep 1-D CNN -
        Conv1d(1->32, k=3, stride=1, ReLU) -> MaxPool1d(2) -> BatchNorm
        -> Dropout(0.15) -> Flatten -> Dense(64, ReLU) -> Dropout
        -> Linear(2) softmax.
        SGD + step-decay LR, early stopping on validation accuracy,
        keep-best-val weights, max 200 epochs. The 320-vector (or the
        PCA-reduced vector) is treated as a length-D, 1-channel signal.
  * Logistic Regression, SVM (RBF), Random Forest, XGBoost - each tuned
        with a small cross-validation grid on the TRAIN rows only.
        class_weight="balanced" (or scale_pos_weight) handles imbalance.

For every (dataset x representation x model x seed) we fit on the TRAIN
core (85 %), early-stop the CNN on the carved validation (15 %), then
predict the probability of the ACTIVE class on the untouched TEST set.
Predictions are cached to results/predictions/ so metrics.py (and the
bootstrap CIs) never need to refit anything.

Run:  python -m src.models                 # everything (slow: ~400 fits)
      python -m src.models toxicity         # one dataset
      python -m src.models toxicity full logreg   # one cell, for debugging
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner, set_all_seeds
from src.embed import load_cache
from src.split import make_train_val
from src.pca_ablation import build_representations

warnings.filterwarnings("ignore")

PRED_DIR = os.path.join(CFG.RESULTS, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

ALL_MODELS = ["cnn", "logreg", "svm", "rf", "xgb"]


# ================================================================== #
# CNN (PyTorch)
# ================================================================== #
def _make_cnn(input_dim):
    import torch.nn as nn

    class UniCNN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.conv = nn.Conv1d(1, CFG.CNN_FILTERS, CFG.CNN_KERNEL, stride=1)
            self.pool = nn.MaxPool1d(2)
            self.bn = nn.BatchNorm1d(CFG.CNN_FILTERS)
            self.drop1 = nn.Dropout(CFG.CNN_DROPOUT)
            # work out the flattened size with the arithmetic:
            l1 = d - CFG.CNN_KERNEL + 1       # after conv (no padding)
            l2 = l1 // 2                       # after maxpool(2)
            self.flat_dim = CFG.CNN_FILTERS * l2
            self.fc1 = nn.Linear(self.flat_dim, CFG.CNN_DENSE)
            self.drop2 = nn.Dropout(CFG.CNN_DROPOUT)
            self.fc2 = nn.Linear(CFG.CNN_DENSE, 2)
            self.relu = nn.ReLU()

        def forward(self, x):                  # x: (B, D)
            x = x.unsqueeze(1)                 # (B, 1, D)  -> 1 channel
            x = self.relu(self.conv(x))
            x = self.pool(x)
            x = self.bn(x)
            x = self.drop1(x)
            x = x.flatten(1)
            x = self.relu(self.fc1(x))
            x = self.drop2(x)
            return self.fc2(x)                 # logits; softmax in loss

    return UniCNN(input_dim)


def train_cnn(Xtr, ytr, Xval, yval, Xte, seed):
    """Train the CNN with early stopping; return P(active) on TEST."""
    import torch
    import torch.nn as nn
    set_all_seeds(seed)
    torch.set_num_threads(CFG.TORCH_THREADS)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xval_t = torch.tensor(Xval, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    # class weights for imbalance (inverse frequency)
    counts = np.bincount(ytr, minlength=2).astype(float)
    w = counts.sum() / (2.0 * np.clip(counts, 1, None))
    class_w = torch.tensor(w, dtype=torch.float32)

    model = _make_cnn(Xtr.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=CFG.CNN_LR, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=CFG.CNN_LR_STEP,
                                            gamma=CFG.CNN_LR_GAMMA)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    n = len(Xtr_t)
    best_val_acc, best_state, patience = -1.0, None, 0
    rng = np.random.RandomState(seed)

    for epoch in range(CFG.CNN_MAX_EPOCH):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, CFG.CNN_BATCH):
            idx = order[s:s + CFG.CNN_BATCH]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()

        # validation accuracy -> early stopping / keep-best
        model.eval()
        with torch.no_grad():
            vpred = model(Xval_t).argmax(1).numpy()
        val_acc = float((vpred == yval).mean())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CFG.CNN_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte_t), dim=1)[:, 1].numpy()
    return prob, best_val_acc


# ================================================================== #
# classical baselines (scikit-learn / xgboost), CV-tuned on TRAIN
# ================================================================== #
def train_classical(kind, Xtr, ytr, Xte, seed):
    """Fit one classical model with a small CV grid on TRAIN; return P(active)."""
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
    best = gs.best_estimator_
    prob = best.predict_proba(Xte)[:, 1]
    return prob, gs.best_params_


# ================================================================== #
# runner
# ================================================================== #
def run_cell(name, repr_name, model, seed, cache, verbose=True):
    """Fit one (dataset, repr, model, seed) and cache test predictions."""
    out_path = os.path.join(
        PRED_DIR, "{}__{}__{}__seed{}.npz".format(name, model, repr_name, seed))
    if os.path.exists(out_path):
        return out_path, None

    X, y, split = cache["X"], cache["y"], cache["split"]
    tr_all = np.where(split == "train")[0]
    te = np.where(split == "test")[0]
    # carve validation from TRAIN only (per seed)
    core_rel, val_rel = make_train_val(y[tr_all], seed)
    tr = tr_all[core_rel]
    val = tr_all[val_rel]

    reps = build_representations(X[tr], X[val], X[te])
    Xtr, Xval, Xte, ncomp = reps[repr_name]
    ytr, yval, yte = y[tr], y[val], y[te]

    t0 = time.time()
    if model == "cnn":
        prob, info = train_cnn(Xtr, ytr, Xval, yval, Xte, seed)
    else:
        prob, info = train_classical(model, Xtr, ytr, Xte, seed)
    dt = time.time() - t0

    np.savez(out_path, y_true=yte, y_prob=prob.astype(np.float32),
             n_comp=ncomp, seconds=dt)
    if verbose:
        print("    {:<12} {:<6} seed{} -> saved ({} comp, {:.1f}s)".format(
            model, repr_name, seed, ncomp, dt))
    return out_path, info


def main():
    banner("models.py - CNN + classical baselines (5 seeds, 4 representations)")
    args = sys.argv[1:]
    datasets = [a for a in args if a in CFG.DATASETS] or CFG.DATASETS
    reprs = [a for a in args if a in CFG.REPRS] or CFG.REPRS
    models = [a for a in args if a in ALL_MODELS] or ALL_MODELS

    for name in datasets:
        print("\n>>> dataset: {}".format(name))
        cache = load_cache(name)
        for repr_name in reprs:
            for model in models:
                for seed in CFG.SEEDS:
                    run_cell(name, repr_name, model, seed, cache)
    print("\n  done. predictions in {}".format(os.path.relpath(PRED_DIR, CFG.ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
