#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/masking.py
==================================================================
The MAIN deliverable: sliding-window masking to show WHICH REGIONS of a
peptide the model depends on (the teacher-required technique).

Idea
----
Take the best CNN on the full 320-D ESM-2 representation. For a test
peptide we:
  1. compute p_full = P(active) for the untouched sequence,
  2. slide a window of length k (k in {3,5,7,9}, stride 1) along the
     sequence; for each window position we MASK that window IN THE
     SEQUENCE (not in the embedding), re-embed with ESM-2, predict, and
     record  delta_p = p_full - p_masked.
     A large positive delta_p means "hiding this region hurt the model's
     confidence a lot" -> that region is model-IMPORTANT.
  3. Two masking styles are used so results don't depend on the trick:
        "X"      : replace the window's letters with the ESM-2 'X' token
                   (unknown residue),
        "delete" : cut the window out of the sequence entirely.
  4. Residue importance = average delta_p of every window covering that
     residue. Peptide summary = max window delta_p and mean of the top-q.

Speed: every masked variant of one peptide is embedded in ONE batched
ESM-2 call, under torch.no_grad(), with an in-memory cache.

The canonical CNN + MinMax scaler used by predict() are trained once per
dataset (seed 0, train core only) and cached to results/masking_models/.

Run:  python -m src.masking                 # all datasets, both styles
      python -m src.masking acp              # one dataset
"""

import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner, set_all_seeds
from src.embed import load_cache, embed_sequences
from src.split import make_train_val
from src.models import _make_cnn

from sklearn.preprocessing import MinMaxScaler

MASK_MODEL_DIR = os.path.join(CFG.RESULTS, "masking_models")
MASK_OUT_DIR = os.path.join(CFG.RESULTS, "masking")
os.makedirs(MASK_MODEL_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)

CANON_SEED = 0   # the single "best CNN" per dataset is trained with this seed


# ================================================================== #
# canonical predictor: one CNN + scaler per dataset
# ================================================================== #
def build_predictor(name, force=False):
    """Train (or load) the canonical full-320 CNN + scaler for a dataset.

    Returns a function predict_batch(seqs)->P(active) and a dict of info.
    """
    import torch
    model_path = os.path.join(MASK_MODEL_DIR, "{}_cnn.pt".format(name))
    scaler_path = os.path.join(MASK_MODEL_DIR, "{}_scaler.pkl".format(name))

    cache = load_cache(name)
    X, y, split = cache["X"], cache["y"], cache["split"]
    tr_all = np.where(split == "train")[0]
    te = np.where(split == "test")[0]
    core_rel, val_rel = make_train_val(y[tr_all], CANON_SEED)
    tr, val = tr_all[core_rel], tr_all[val_rel]

    scaler = MinMaxScaler().fit(X[tr])           # TRAIN only

    if os.path.exists(model_path) and os.path.exists(scaler_path) and not force:
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        model = _make_cnn(CFG.EMB_DIM)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        # train the canonical CNN (same routine as models.py) and save it.
        # _train_cnn_keep mirrors models.train_cnn but returns the fitted model.
        set_all_seeds(CANON_SEED)
        Str, Sval = scaler.transform(X[tr]), scaler.transform(X[val])
        model = _train_cnn_keep(Str, y[tr], Sval, y[val], CANON_SEED)
        torch.save(model.state_dict(), model_path)
        with open(scaler_path, "wb") as fh:
            pickle.dump(scaler, fh)

    def predict_batch(seqs):
        """Raw sequences -> P(active). Embeds, scales with TRAIN scaler, CNN."""
        if isinstance(seqs, str):
            seqs = [seqs]
        emb = embed_sequences(seqs)              # (N,320)
        Z = scaler.transform(emb)
        with torch.no_grad():
            logits = model(torch.tensor(Z, dtype=torch.float32))
            prob = torch.softmax(logits, dim=1)[:, 1].numpy()
        return prob

    info = {"test_idx": te, "sequence": cache["sequence"], "y": y}
    return predict_batch, info


def _train_cnn_keep(Xtr, ytr, Xval, yval, seed):
    """Same as models.train_cnn but RETURNS the fitted model (not preds)."""
    import torch
    import torch.nn as nn
    set_all_seeds(seed)
    torch.set_num_threads(CFG.TORCH_THREADS)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xval_t = torch.tensor(Xval, dtype=torch.float32)
    counts = np.bincount(ytr, minlength=2).astype(float)
    w = counts.sum() / (2.0 * np.clip(counts, 1, None))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

    model = _make_cnn(Xtr.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=CFG.CNN_LR, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, CFG.CNN_LR_STEP, CFG.CNN_LR_GAMMA)
    n = len(Xtr_t)
    best_acc, best_state, patience = -1.0, None, 0
    rng = np.random.RandomState(seed)
    for epoch in range(CFG.CNN_MAX_EPOCH):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, CFG.CNN_BATCH):
            idx = order[s:s + CFG.CNN_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vacc = float((model(Xval_t).argmax(1).numpy() == yval).mean())
        if vacc > best_acc:
            best_acc, best_state, patience = vacc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= CFG.CNN_PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ================================================================== #
# masking variants of one peptide
# ================================================================== #
def window_variants(seq, k, style):
    """Yield (start, masked_sequence) for every window of length k.

    style 'X'      -> window letters replaced by 'X'
    style 'delete' -> window cut out
    Peptides shorter than k produce NO windows (handled by caller).
    """
    L = len(seq)
    out = []
    for i in range(0, L - k + 1):
        if style == "X":
            masked = seq[:i] + ("X" * k) + seq[i + k:]
        elif style == "delete":
            masked = seq[:i] + seq[i + k:]
        else:
            raise ValueError("unknown style " + style)
        out.append((i, masked))
    return out


def analyse_peptide(seq, predict_batch, p_full, cache):
    """Return per-k window deltas + residue importance for one peptide.

    cache : dict seq->prob (in-memory) to avoid re-embedding repeats.
    """
    L = len(seq)
    per_k = {}            # k -> list of (start, delta_p)
    # residue importance accumulators (averaged over ALL windows, all k)
    res_sum = np.zeros(L)
    res_cnt = np.zeros(L)

    for k in CFG.MASK_WINDOWS:
        if L < k:
            per_k[k] = []
            continue
        variants = window_variants(seq, k, analyse_peptide.style)
        # batch all variants for this k in one embed call (speed)
        to_embed = [m for _, m in variants if m not in cache]
        if to_embed:
            probs = predict_batch(to_embed)
            for m, p in zip(to_embed, probs):
                cache[m] = float(p)
        deltas = []
        for (start, m) in variants:
            dp = p_full - cache[m]
            deltas.append((start, dp))
            res_sum[start:start + k] += dp
            res_cnt[start:start + k] += 1
        per_k[k] = deltas

    with np.errstate(invalid="ignore"):
        residue_importance = np.where(res_cnt > 0, res_sum / np.maximum(res_cnt, 1), 0.0)

    # peptide-level summary from the finest-grained windows available
    all_deltas = [dp for k in CFG.MASK_WINDOWS for (_, dp) in per_k.get(k, [])]
    if all_deltas:
        arr = np.sort(np.array(all_deltas))[::-1]
        max_delta = float(arr[0])
        topq_mean = float(arr[:CFG.MASK_TOPQ].mean())
    else:
        max_delta = topq_mean = float("nan")

    return {
        "seq": seq, "L": L, "p_full": float(p_full),
        "per_k": per_k, "residue_importance": residue_importance,
        "max_delta": max_delta, "topq_mean": topq_mean,
    }


# ================================================================== #
# run masking over a dataset
# ================================================================== #
def run_dataset(name, style, which="active"):
    """Run masking on test peptides the CNN correctly calls ACTIVE (default).

    which: 'active'   -> true-positive active peptides (correctly predicted)
           'inactive' -> correctly predicted inactive (for contrast)
    """
    banner("masking {} | style={} | subset={}".format(name, style, which))
    out = os.path.join(MASK_OUT_DIR, "{}__{}__{}.pkl".format(name, style, which))
    if os.path.exists(out):
        print("  already done, skipping: {}".format(os.path.basename(out)))
        with open(out, "rb") as fh:
            return pickle.load(fh)
    predict_batch, info = build_predictor(name)
    analyse_peptide.style = style

    te = info["test_idx"]
    seqs = info["sequence"][te]
    ylab = info["y"][te]

    # predict full-sequence probabilities for the test set (batched)
    p_full_all = predict_batch(list(seqs))
    pred_pos = p_full_all >= 0.5

    if which == "active":
        keep = (ylab == 1) & pred_pos            # correctly predicted active
    else:
        keep = (ylab == 0) & (~pred_pos)         # correctly predicted inactive

    idx = np.where(keep)[0]
    print("  {} peptides selected ({} / {} test)".format(len(idx), len(idx), len(te)))

    cache = {}
    results = []
    for n_done, j in enumerate(idx):
        seq = str(seqs[j])
        if len(seq) < min(CFG.MASK_WINDOWS):
            continue                              # too short to mask
        r = analyse_peptide(seq, predict_batch, float(p_full_all[j]), cache)
        r["label"] = int(ylab[j])
        results.append(r)
        if (n_done + 1) % 25 == 0:
            print("    analysed {}/{}  (cache={})".format(n_done + 1, len(idx), len(cache)))

    out = os.path.join(MASK_OUT_DIR, "{}__{}__{}.pkl".format(name, style, which))
    with open(out, "wb") as fh:
        pickle.dump(results, fh)
    print("  wrote {}  ({} peptides)".format(os.path.relpath(out, CFG.ROOT), len(results)))
    return results


def main():
    datasets = [a for a in sys.argv[1:] if a in CFG.DATASETS] or CFG.DATASETS
    for name in datasets:
        for style in CFG.MASK_STYLES:
            run_dataset(name, style, which="active")
    print("\n  masking done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
