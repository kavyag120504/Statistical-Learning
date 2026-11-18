#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/split.py
==================================================================
Leakage control + splitting for the UniDL4BioPep project.

The clean CSVs in data/clean/ already carry the paper's official
train/test split and have 0 EXACT train/test overlap (done in
preprocess_dataset.py). This script adds the two things the project
brief still asks for:

  1. NEAR-identical train/test check. Exact duplicates are gone, but a
     test peptide that is ~the same as a train peptide still leaks. We
     measure 3-mer Jaccard similarity between every test sequence and
     the train sequences of similar length, and DROP from the test set
     any test peptide whose best match to train is >= THRESHOLD. We
     print how many were removed (honest reporting).

  2. A VALIDATION carve. ~15 % of TRAIN is held out for tuning / early
     stopping. It is stratified by label and depends on the seed, so the
     5 seeds get 5 different val splits and we get real variance. This is
     a helper (make_train_val) called at model time - nothing here is fit
     on validation or test.

Output: data/splits/<dataset>.csv  (clean rows, near-dup test removed,
columns: dataset, sequence, length, label, split) + a printed summary
and results/split_summary.csv.

The label orientation (label==1 means "active") is verified/printed per
dataset so the README can state it plainly.

Run:  python -m src.split
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner

NEARDUP_THRESHOLD = 0.90   # 3-mer Jaccard >= this => treated as near-identical
KMER = 3


# ------------------------------------------------------------------ #
# near-duplicate detection (cheap, dependency-free)
# ------------------------------------------------------------------ #
def _kmers(seq, k=KMER):
    """Set of k-mers for one sequence (whole seq if shorter than k)."""
    if len(seq) < k:
        return {seq}
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def _jaccard(a, b):
    """Jaccard similarity of two k-mer sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a) + len(b) - inter)


def find_near_dups(train_seqs, test_seqs, threshold=NEARDUP_THRESHOLD):
    """Return a boolean mask over test_seqs: True == near-duplicate of train.

    To stay fast we only compare a test peptide against train peptides
    whose length is within +/-20 % (near-identical sequences must have
    similar length). This cuts the work from all-pairs to a small slice.
    """
    train_km = [(_kmers(s), len(s)) for s in train_seqs]
    # bucket train indices by length for a quick length-window lookup
    by_len = {}
    for i, (_, L) in enumerate(train_km):
        by_len.setdefault(L, []).append(i)
    max_len = max(by_len) if by_len else 0

    flags = np.zeros(len(test_seqs), dtype=bool)
    for j, s in enumerate(test_seqs):
        ks = _kmers(s)
        L = len(s)
        lo, hi = int(L * 0.8), int(L * 1.2) + 1
        best = 0.0
        for Ln in range(max(1, lo), min(max_len, hi) + 1):
            for i in by_len.get(Ln, ()):
                sim = _jaccard(ks, train_km[i][0])
                if sim > best:
                    best = sim
                    if best >= threshold:
                        break
            if best >= threshold:
                break
        flags[j] = best >= threshold
    return flags


# ------------------------------------------------------------------ #
# validation carve (used by the model scripts, per seed)
# ------------------------------------------------------------------ #
def make_train_val(labels, seed, val_frac=CFG.VAL_FRACTION):
    """Stratified train/val split of the TRAIN rows only.

    Parameters
    ----------
    labels : 1-D array of 0/1 for the TRAIN rows (in row order).
    seed   : int, so each seed gives a different (reproducible) split.

    Returns (train_idx, val_idx) : integer index arrays into the TRAIN rows.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    val_idx = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        val_idx.extend(idx[:n_val].tolist())
    val_idx = np.array(sorted(val_idx))
    mask = np.ones(len(labels), dtype=bool)
    mask[val_idx] = False
    train_idx = np.where(mask)[0]
    return train_idx, val_idx


# ------------------------------------------------------------------ #
# main: run near-dup removal + write per-dataset split files
# ------------------------------------------------------------------ #
def process_dataset(name):
    path = os.path.join(CFG.DATA_CLEAN, "{}.csv".format(name))
    df = pd.read_csv(path)

    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)

    # --- label orientation / counts (verify 1 == active assumption) ---
    print("  label counts (1 == active):")
    for split_name, part in (("train", tr), ("test", te)):
        pos = int((part["label"] == 1).sum())
        neg = int((part["label"] == 0).sum())
        print("    {:<5} pos(1)={:<5} neg(0)={:<5} pos%={:.1f}"
              .format(split_name, pos, neg, 100.0 * pos / max(1, len(part))))

    # --- near-duplicate test removal ---
    flags = find_near_dups(tr["sequence"].tolist(), te["sequence"].tolist())
    n_removed = int(flags.sum())
    te_clean = te[~flags].reset_index(drop=True)
    print("  near-dup test peptides removed (3-mer Jaccard >= {}): {} / {}"
          .format(NEARDUP_THRESHOLD, n_removed, len(te)))

    out = pd.concat([tr, te_clean], ignore_index=True)
    out_path = os.path.join(CFG.DATA_SPLIT, "{}.csv".format(name))
    out.to_csv(out_path, index=False)
    print("  wrote {}  (train={}, test={})".format(
        os.path.relpath(out_path, CFG.ROOT), len(tr), len(te_clean)))

    return {
        "dataset": name,
        "train": len(tr), "test_before": len(te),
        "neardup_removed": n_removed, "test_after": len(te_clean),
        "train_pos%": round(100.0 * (tr["label"] == 1).mean(), 1),
        "test_pos%": round(100.0 * (te_clean["label"] == 1).mean(), 1),
    }


def main():
    banner("split.py - near-dup check + validation carve setup")
    rows = []
    for name in CFG.DATASETS:
        print("\n>>> {}".format(name))
        rows.append(process_dataset(name))

    summ = pd.DataFrame(rows)
    out = os.path.join(CFG.RESULTS, "split_summary.csv")
    summ.to_csv(out, index=False)
    banner("SPLIT SUMMARY")
    print(summ.to_string(index=False))
    print("\n  wrote {}".format(os.path.relpath(out, CFG.ROOT)))
    print("  (validation is carved per-seed at model time via make_train_val)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
