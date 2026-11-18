#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_dataset.py
==================================================================
DSC4804 Statistical Learning  -  UniDL4BioPep project (Phase 1: Data)

PREPARATION ONLY. This script just assembles the raw peptide benchmark
files downloaded from the UniDL4BioPep repository
(https://github.com/dzjxzyd/UniDL4BioPep) into 4 CSV files - one per
bioactivity dataset:

    data/processed/ace.csv           ACE-inhibitory activity
    data/processed/acp.csv           Anticancer peptides (ACP, main set)
    data/processed/neuropeptide.csv  Neuropeptides (NeuroPred)
    data/processed/toxicity.csv      Toxicity (2021 dataset)

It does NOT do any preprocessing/cleaning (no de-duplication, no amino
acid filtering, no label fixing). Every row from the source files is
kept exactly as-is. Cleaning and feature engineering happen in later
steps.

Each output CSV has one uniform schema so later steps can load any
dataset the same way:

    dataset    short dataset id (e.g. "ace")
    sequence   amino-acid sequence, exactly as in the source file
    length     number of characters in the sequence
    label      class label, exactly as in the source file (1=active/0=inactive)
    split      "train" or "test"  (the paper's original split is preserved)

How to use
----------
1. Put the raw files under data/raw/<dataset>/ (see the RAW file names
   in the DATASETS config below). If a file is missing the script just
   WARNS and skips that dataset - it never crashes.
2. Run:  python prepare_dataset.py
3. Read the validation summary printed at the end.
"""

import os
import sys
import glob

import pandas as pd

# ------------------------------------------------------------------ #
# Paths (all relative to this script, so it runs from anywhere)
# ------------------------------------------------------------------ #
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw")
OUT_DIR = os.path.join(HERE, "data", "processed")

# ------------------------------------------------------------------ #
# Dataset configuration
# ------------------------------------------------------------------ #
# For each output dataset we list its source files as (relative_path,
# split) pairs. Splits mirror the UniDL4BioPep train/test partition so
# results stay comparable with the base paper. `url` is only used to
# print a helpful hint when a file is missing.
DATASETS = {
    "ace": {
        "name": "ACE-inhibitory activity",
        "sources": [
            ("ace/AHT_train.xlsx", "train"),
            ("ace/AHT_test.xlsx", "test"),
        ],
        "url": "https://github.com/dzjxzyd/UniDL4BioPep -> '1. ACE inhibitory activity'",
    },
    "acp": {
        "name": "Anticancer peptides (ACP, main)",
        "sources": [
            ("acp/main_ACP_train.xlsx", "train"),
            ("acp/main_ACP_test.xlsx", "test"),
        ],
        "url": "https://github.com/dzjxzyd/UniDL4BioPep -> '8. ACP Anticancer activity-main'",
    },
    "neuropeptide": {
        "name": "Neuropeptides (NeuroPred)",
        "sources": [
            ("neuropeptide/neuro_train.xlsx", "train"),
            ("neuropeptide/neuro_test.xlsx", "test"),
        ],
        "url": "https://github.com/dzjxzyd/UniDL4BioPep -> '13.NeuroPred'",
    },
    "toxicity": {
        "name": "Toxicity (2021 dataset)",
        "sources": [
            ("toxicity/train_3284.xlsx", "train"),
            ("toxicity/test_580.xlsx", "test"),
        ],
        "url": "https://github.com/dzjxzyd/UniDL4BioPep -> '17. Toxicity 2021 Dataset'",
    },
    # --- Optional 5th "validation" dataset from the project charter. ---
    # It ships as a single file with no official train/test split, so we
    # leave it out of the 4 core deliverables. Uncomment to also build it;
    # rows are assigned split="train".
    # "antioxidant": {
    #     "name": "Antioxidant activity (FRS)",
    #     "sources": [("antioxidant/antioxidant.xlsx", "train")],
    #     "url": "https://github.com/dzjxzyd/UniDL4BioPep -> '18. antioxidant_FRS'",
    # },
}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _find_columns(df):
    """Locate the sequence and label columns regardless of casing/spacing."""
    seq_col = label_col = None
    for c in df.columns:
        key = str(c).strip().lower()
        if seq_col is None and key in ("sequence", "seq", "peptide", "sequences"):
            seq_col = c
        elif label_col is None and key in ("label", "labels", "y", "class", "target"):
            label_col = c
    # Fallback: assume first col = sequence, second = label (repo default).
    if seq_col is None and len(df.columns) >= 1:
        seq_col = df.columns[0]
    if label_col is None and len(df.columns) >= 2:
        label_col = df.columns[1]
    return seq_col, label_col


def _load_source(path, split):
    """Read one raw file and return a DataFrame (sequence, label, split).

    Rows are taken exactly as they appear in the source - no cleaning.
    Returns None if the file is unreadable or has no usable columns.
    """
    try:
        df = pd.read_excel(path)
    except Exception as exc:  # unreadable / corrupt file
        print("    [!] could not read {}: {}".format(os.path.basename(path), exc))
        return None

    seq_col, label_col = _find_columns(df)
    if seq_col is None or label_col is None:
        print("    [!] {} has no sequence/label columns (found {}) - skipping"
              .format(os.path.basename(path), list(df.columns)))
        return None

    out = df[[seq_col, label_col]].copy()
    out.columns = ["sequence", "label"]
    out["split"] = split
    return out


def build_dataset(key, cfg):
    """Assemble one dataset CSV. Returns (DataFrame or None, missing_files)."""
    frames = []
    missing = []
    for rel_path, split in cfg["sources"]:
        full = os.path.join(RAW_DIR, rel_path)
        if not os.path.exists(full):
            missing.append(rel_path)
            print("    [skip] missing source: {}".format(rel_path))
            continue
        print("    reading {} (split={})".format(rel_path, split))
        part = _load_source(full, split)
        if part is not None and len(part):
            frames.append(part)

    if not frames:
        return None, missing

    df = pd.concat(frames, ignore_index=True)

    # Assemble the uniform schema (no cleaning of values).
    df["dataset"] = key
    df["length"] = df["sequence"].astype(str).str.len()
    df = df[["dataset", "sequence", "length", "label", "split"]]
    return df, missing


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    print("=" * 70)
    print("  UniDL4BioPep - dataset preparation (assembly only, no cleaning)")
    print("  raw in : {}".format(RAW_DIR))
    print("  out to : {}".format(OUT_DIR))
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isdir(RAW_DIR) or not glob.glob(os.path.join(RAW_DIR, "*", "*")):
        print("\n[!] No raw files found under data/raw/.")
        print("    Download the datasets from the UniDL4BioPep repo and place")
        print("    each dataset's .xlsx files under data/raw/<dataset>/.")
        # Not a crash: we still print an (empty) summary below.

    summaries = []
    for key, cfg in DATASETS.items():
        print("\n>>> {}  ({})".format(key, cfg["name"]))
        try:
            df, missing = build_dataset(key, cfg)
        except Exception as exc:  # never let one dataset kill the run
            print("    [!] unexpected error building '{}': {} - skipping".format(key, exc))
            continue

        if df is None or len(df) == 0:
            print("    [warn] no usable data - dataset skipped.")
            print("           expected source: {}".format(cfg["url"]))
            summaries.append({"dataset": key, "status": "SKIPPED", "rows": 0})
            continue

        out_path = os.path.join(OUT_DIR, "{}.csv".format(key))
        df.to_csv(out_path, index=False)

        n = len(df)
        n_train = int((df["split"] == "train").sum())
        n_test = int((df["split"] == "test").sum())
        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())
        print("    wrote {}  ({} rows)".format(os.path.relpath(out_path, HERE), n))

        summaries.append({
            "dataset": key, "status": "OK", "rows": n,
            "train": n_train, "test": n_test,
            "pos": n_pos, "neg": n_neg,
            "pos_pct": round(100.0 * n_pos / n, 1) if n else 0.0,
            "len_min": int(df["length"].min()),
            "len_med": int(df["length"].median()),
            "len_max": int(df["length"].max()),
        })

    # ------------------------- final summary -------------------------- #
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    ok = [s for s in summaries if s["status"] == "OK"]
    if ok:
        hdr = ("{:<13} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>13}"
               .format("dataset", "rows", "train", "test", "pos", "neg", "pos%", "len(min/med/max)"))
        print(hdr)
        print("-" * len(hdr))
        for s in ok:
            print("{:<13} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6.1f}% {:>5}/{:>3}/{:>3}".format(
                s["dataset"], s["rows"], s["train"], s["test"], s["pos"], s["neg"],
                s["pos_pct"], s["len_min"], s["len_med"], s["len_max"]))
        total = sum(s["rows"] for s in ok)
        print("-" * len(hdr))
        print("{:<13} {:>6}".format("TOTAL", total))

    skipped = [s for s in summaries if s["status"] != "OK"]
    if skipped:
        print("\n  Skipped datasets (missing/unusable sources): {}".format(
            ", ".join(s["dataset"] for s in skipped)))

    print("\n  {} of {} datasets prepared -> {}".format(
        len(ok), len(DATASETS), OUT_DIR))
    print("  Done. (raw rows kept as-is; cleaning/preprocessing comes later)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
