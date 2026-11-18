#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_dataset.py
==================================================================
DSC4804 Statistical Learning  -  UniDL4BioPep project

Cleans the assembled CSVs produced by prepare_dataset.py and writes
analysis-ready, per-dataset cleaned files:

    data/processed/<name>.csv   (input: assembled, raw rows)
        -> data/clean/<name>.csv   (output: cleaned)

The 4 datasets stay as 4 SEPARATE files - they are 4 different
classification tasks (ACE-inhibition, anticancer, neuropeptide,
toxicity) and must NOT be merged. Each keeps its own train/test split.

Preprocessing applied (standard, conservative, per dataset):
  1. Normalise sequence  : strip whitespace, upper-case.
  2. Drop empty sequences : blank / NaN rows.
  3. Drop invalid residues: sequences containing anything other than the
                            20 standard amino acids (ACDEFGHIKLMNPQRSTVWY).
                            Non-standard letters (B J O U X Z), gaps, digits
                            etc. are not usable for descriptors or ESM-2.
  4. Fix labels           : coerce to int; drop anything that is not 0/1.
  5. Drop label conflicts : if the same sequence appears with BOTH labels
                            the ground truth is ambiguous -> remove all copies.
  6. De-duplicate         : identical sequences collapsed to one row. When a
                            sequence appears in both splits, the train copy is
                            kept and the test copy removed, so the test set
                            stays a clean hold-out (no train/test leakage).

Schema is unchanged: dataset, sequence, length, label, split.

Run:  python preprocess_dataset.py
"""

import os
import sys
import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "data", "processed")   # assembled CSVs
OUT_DIR = os.path.join(HERE, "data", "clean")       # cleaned CSVs

# The 20 standard amino acids.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_one(path):
    """Clean a single assembled CSV. Returns (clean_df, report dict)."""
    df = pd.read_csv(path)
    rep = {
        "in": len(df), "empty": 0, "bad_aa": 0, "bad_label": 0,
        "conflict": 0, "dup": 0,
    }

    # 1. normalise sequence
    df["sequence"] = (
        df["sequence"].astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True)
    )

    # 2. drop empty
    empty = df["sequence"].isin(["", "NAN", "NONE"])
    rep["empty"] = int(empty.sum())
    df = df[~empty]

    # 3. drop invalid residues
    valid = df["sequence"].apply(lambda s: set(s) <= STANDARD_AA)
    rep["bad_aa"] = int((~valid).sum())
    df = df[valid]

    # 4. fix labels -> int 0/1
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    bad = ~df["label"].isin([0, 1])
    rep["bad_label"] = int(bad.sum())
    df = df[~bad]
    df["label"] = df["label"].astype(int)

    # 5. drop label conflicts (same sequence, both labels)
    conflict = (
        df.groupby("sequence")["label"].nunique().loc[lambda s: s > 1].index
    )
    if len(conflict):
        rep["conflict"] = int(df["sequence"].isin(conflict).sum())
        df = df[~df["sequence"].isin(conflict)]

    # 6. de-duplicate, keeping the train copy over test (clean hold-out)
    rank = {"train": 0, "test": 1}
    df = df.assign(_r=df["split"].map(rank).fillna(9))
    before = len(df)
    df = (
        df.sort_values("_r")
          .drop_duplicates(subset=["sequence"], keep="first")
          .drop(columns="_r")
    )
    rep["dup"] = before - len(df)

    # recompute length and tidy ordering
    df["length"] = df["sequence"].str.len()
    df = df[["dataset", "sequence", "length", "label", "split"]]
    df = df.sort_values(["split", "label", "sequence"]).reset_index(drop=True)
    rep["out"] = len(df)
    return df, rep


def main():
    print("=" * 70)
    print("  UniDL4BioPep - preprocessing (clean, per dataset, kept separate)")
    print("  in : {}".format(IN_DIR))
    print("  out: {}".format(OUT_DIR))
    print("=" * 70)

    inputs = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not inputs:
        print("\n[!] No assembled CSVs in data/processed/.")
        print("    Run prepare_dataset.py first.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = []

    for path in inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        print("\n>>> {}".format(name))
        try:
            df, rep = clean_one(path)
        except Exception as exc:
            print("    [!] error cleaning {}: {} - skipping".format(name, exc))
            continue

        out_path = os.path.join(OUT_DIR, "{}.csv".format(name))
        df.to_csv(out_path, index=False)
        removed = rep["in"] - rep["out"]
        print("    {in} -> {out} rows  (removed {removed}: empty={empty}, "
              "bad_aa={bad_aa}, bad_label={bad_label}, conflict={conflict}, dup={dup})"
              .format(removed=removed, **rep))
        print("    wrote {}".format(os.path.relpath(out_path, HERE)))

        n = len(df)
        summaries.append({
            "dataset": name, "rows": n,
            "train": int((df["split"] == "train").sum()),
            "test": int((df["split"] == "test").sum()),
            "pos": int((df["label"] == 1).sum()),
            "neg": int((df["label"] == 0).sum()),
            "pos_pct": round(100.0 * int((df["label"] == 1).sum()) / n, 1) if n else 0.0,
            "len_min": int(df["length"].min()), "len_med": int(df["length"].median()),
            "len_max": int(df["length"].max()),
        })

    # ------------------------- final summary -------------------------- #
    print("\n" + "=" * 70)
    print("  CLEANED DATA SUMMARY")
    print("=" * 70)
    if summaries:
        hdr = ("{:<13} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>13}"
               .format("dataset", "rows", "train", "test", "pos", "neg", "pos%", "len(min/med/max)"))
        print(hdr)
        print("-" * len(hdr))
        for s in summaries:
            print("{:<13} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6.1f}% {:>5}/{:>3}/{:>3}".format(
                s["dataset"], s["rows"], s["train"], s["test"], s["pos"], s["neg"],
                s["pos_pct"], s["len_min"], s["len_med"], s["len_max"]))
        print("-" * len(hdr))
        print("{:<13} {:>6}".format("TOTAL", sum(s["rows"] for s in summaries)))

    print("\n  {} datasets cleaned -> {}".format(len(summaries), OUT_DIR))
    print("  Kept as separate files (do NOT merge - 4 distinct tasks).")
    print("  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
