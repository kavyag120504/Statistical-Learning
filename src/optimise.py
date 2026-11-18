#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/optimise.py  (OPTIONAL extension)
==================================================================
Simple guided mutation ("peptide optimisation") to demonstrate what the
CNN *prefers*. This does NOT claim the designed peptides are truly more
active - only that the model scores them higher.

Procedure (simulated-annealing style), per starting peptide, 5 seeds:
  * compute residue importance (via masking) once for the start peptide,
  * each round: pick a position with probability proportional to its
    importance, replace its amino acid with a random one (weighted by the
    amino-acid frequency in ACTIVE training peptides),
  * accept the mutant if P(active) improves; otherwise accept it anyway
    with a small probability (so the search does not get stuck),
  * run OPT_ROUNDS rounds, keep the best peptide found.

Cross-check: the best peptides are re-scored with a RANDOM FOREST trained
on the same embeddings. If the RF also scores them higher, the gain is
less likely to be the CNN being fooled.

Run:  python -m src.optimise acp
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner, set_all_seeds
from src.embed import load_cache, embed_sequences
from src.split import make_train_val
from src.masking import build_predictor, analyse_peptide, MASK_OUT_DIR

AA = "ACDEFGHIKLMNPQRSTVWY"


def active_aa_frequencies(name):
    """Amino-acid frequencies among ACTIVE training peptides (for sampling)."""
    cache = load_cache(name)
    seqs = cache["sequence"][(cache["split"] == "train") & (cache["y"] == 1)]
    counts = {a: 1 for a in AA}                 # +1 smoothing
    for s in seqs:
        for a in str(s):
            if a in counts:
                counts[a] += 1
    tot = sum(counts.values())
    return np.array([counts[a] / tot for a in AA])


def rf_crosscheck_model(name):
    """Train a Random Forest on full embeddings for an independent re-score."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    cache = load_cache(name)
    X, y, split = cache["X"], cache["y"], cache["split"]
    tr_all = np.where(split == "train")[0]
    core_rel, _ = make_train_val(y[tr_all], 0)
    tr = tr_all[core_rel]
    scaler = MinMaxScaler().fit(X[tr])
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                random_state=0, n_jobs=-1).fit(scaler.transform(X[tr]), y[tr])

    def score(seqs):
        emb = embed_sequences(list(seqs))
        return rf.predict_proba(scaler.transform(emb))[:, 1]
    return score


def optimise_peptide(seq, predict_batch, aa_freq, seed, rounds=CFG.OPT_ROUNDS,
                     temp=0.1):
    """Simulated-annealing mutation of one peptide. Returns (best_seq, trace)."""
    set_all_seeds(seed)
    rng = np.random.RandomState(seed)

    # residue importance of the START peptide -> position sampling weights
    analyse_peptide.style = "X"
    rec = analyse_peptide(seq, predict_batch, float(predict_batch([seq])[0]), {})
    imp = np.clip(rec["residue_importance"], 0, None)
    if imp.sum() <= 0:
        imp = np.ones(len(seq))
    pos_w = imp / imp.sum()

    cur = list(seq)
    cur_p = float(predict_batch(["".join(cur)])[0])
    best, best_p = list(cur), cur_p
    trace = [cur_p]
    for _ in range(rounds):
        i = int(rng.choice(len(cur), p=pos_w))
        new_aa = AA[int(rng.choice(len(AA), p=aa_freq))]
        cand = list(cur)
        cand[i] = new_aa
        cand_p = float(predict_batch(["".join(cand)])[0])
        if cand_p >= cur_p or rng.rand() < np.exp((cand_p - cur_p) / temp):
            cur, cur_p = cand, cand_p
            if cur_p > best_p:
                best, best_p = list(cur), cur_p
        trace.append(cur_p)
    return "".join(best), best_p, trace


def run_dataset(name, n_start=5):
    banner("peptide optimisation (OPTIONAL) - {}".format(name))
    predict_batch, info = build_predictor(name)
    aa_freq = active_aa_frequencies(name)

    # start from correctly-predicted active test peptides that have HEADROOM:
    # the lowest-confidence correct actives (p_full still >= 0.5). The most
    # confident peptides already sit at p=1.0 and cannot be pushed higher, so
    # starting there makes the optimisation vacuous (zero gain). Choosing the
    # least-confident correct actives gives the search room to climb and shows
    # whether importance-guided mutation actually raises the predicted activity.
    mpath = os.path.join(MASK_OUT_DIR, "{}__X__active.pkl".format(name))
    if os.path.exists(mpath):
        with open(mpath, "rb") as fh:
            recs = pickle.load(fh)
        starts = [r["seq"] for r in sorted(recs, key=lambda r: r["p_full"])[:n_start]]
    else:
        te = info["test_idx"]
        seqs = info["sequence"][te]
        y = info["y"][te]
        p = predict_batch(list(seqs))
        correct = [(str(s), pp) for s, yy, pp in zip(seqs, y, p) if yy == 1 and pp >= 0.5]
        starts = [s for s, _ in sorted(correct, key=lambda x: x[1])[:n_start]]

    rf_score = rf_crosscheck_model(name)
    rows = []
    for si, s in enumerate(starts):
        for seed in CFG.OPT_SEEDS:
            best, best_p, _ = optimise_peptide(s, predict_batch, aa_freq, seed)
            start_p = float(predict_batch([s])[0])
            rf_start = float(rf_score([s])[0])
            rf_best = float(rf_score([best])[0])
            rows.append({
                "dataset": name, "start_idx": si, "seed": seed,
                "start_seq": s, "best_seq": best,
                "cnn_p_start": round(start_p, 4), "cnn_p_best": round(best_p, 4),
                "cnn_gain": round(best_p - start_p, 4),
                "rf_p_start": round(rf_start, 4), "rf_p_best": round(rf_best, 4),
                "rf_gain": round(rf_best - rf_start, 4),
                "n_mutations": sum(a != b for a, b in zip(s, best)),
            })
            print("  start{} seed{}: CNN {:.3f}->{:.3f}  RF {:.3f}->{:.3f}".format(
                si, seed, start_p, best_p, rf_start, rf_best))

    df = pd.DataFrame(rows)
    out = os.path.join(CFG.RESULTS, "optimise_{}.csv".format(name))
    df.to_csv(out, index=False)
    print("\n  CNN mean gain {:.3f}, RF mean gain {:.3f} (RF is the independent check)"
          .format(df["cnn_gain"].mean(), df["rf_gain"].mean()))
    print("  wrote {}".format(os.path.relpath(out, CFG.ROOT)))
    return df


def main():
    datasets = sys.argv[1:] or CFG.DATASETS
    for name in datasets:
        run_dataset(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
