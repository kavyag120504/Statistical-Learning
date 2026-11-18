#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/validate.py
==================================================================
Does the masking result mean anything? This script runs the four
required checks on the masking output (results/masking/*.pkl):

  (a) DELETION TEST  - for each peptide, mask the top-m ranked windows,
      then m RANDOM windows, then the bottom-m windows, and measure the
      drop in P(active). If the method works, masking the top windows
      should drop the probability much more than random/bottom. We use a
      paired Wilcoxon signed-rank test (top vs random) and report an
      effect size (rank-biserial correlation + mean drop).

  (b) STABILITY  - Spearman correlation of the residue-importance profile
      across window sizes k, and across the two masking styles (X vs
      delete). High correlation = the "important regions" are not an
      artefact of one particular setting.

  (c) COMPOSITION - are amino acids (K, R, L, W, C, ...) enriched in the
      top windows vs the rest of the peptide? Reported as a frequency
      ratio per residue with a chi-square style log2 enrichment.

  (d) POSITION    - average residue importance by RELATIVE position
      (N-terminus -> C-terminus, 10 bins), aggregated over all active
      peptides. Saved for the position figure.

All numbers are written to results/ as CSVs. Nothing is invented.

Run:  python -m src.validate              # all datasets
      python -m src.validate acp
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner
from src.masking import build_predictor, MASK_OUT_DIR, window_variants

AA = "ACDEFGHIKLMNPQRSTVWY"


def load_masking(name, style, which="active"):
    path = os.path.join(MASK_OUT_DIR, "{}__{}__{}.pkl".format(name, style, which))
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------------ #
# residue importance for a single k (from stored per-k window deltas)
# ------------------------------------------------------------------ #
def residue_importance_for_k(rec, k):
    L = rec["L"]
    s = np.zeros(L)
    c = np.zeros(L)
    for (start, dp) in rec["per_k"].get(k, []):
        s[start:start + k] += dp
        c[start:start + k] += 1
    return np.where(c > 0, s / np.maximum(c, 1), 0.0)


# ------------------------------------------------------------------ #
# (a) deletion test
# ------------------------------------------------------------------ #
def deletion_test(name, style="X", k=5, m=CFG.DELETION_M, max_pep=120, seed=0):
    recs = load_masking(name, style)
    if not recs:
        return None
    predict_batch, _ = build_predictor(name)
    rng = np.random.RandomState(seed)

    rows = []
    recs = [r for r in recs if len(r["per_k"].get(k, [])) >= m]
    recs = recs[:max_pep]                       # cap for speed
    for r in recs:
        seq, L, p_full = r["seq"], r["L"], r["p_full"]
        deltas = r["per_k"][k]                   # list of (start, dp)
        order = sorted(deltas, key=lambda x: x[1], reverse=True)
        top = [s for s, _ in order[:m]]
        bottom = [s for s, _ in order[-m:]]
        randoms = [s for s, _ in [deltas[i] for i in rng.choice(len(deltas), m, replace=False)]]

        def mask_union(starts):
            masked = list(seq)
            for st in starts:
                for p in range(st, min(st + k, L)):
                    masked[p] = "X"
            return "".join(masked)

        variants = [mask_union(top), mask_union(randoms), mask_union(bottom)]
        p_top, p_rand, p_bot = predict_batch(variants)
        rows.append({"seq": seq, "p_full": p_full,
                     "drop_top": p_full - p_top,
                     "drop_rand": p_full - p_rand,
                     "drop_bottom": p_full - p_bot})

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # paired Wilcoxon: top vs random
    try:
        w_stat, w_p = wilcoxon(df["drop_top"], df["drop_rand"])
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    # rank-biserial effect size from Wilcoxon (how often top > random)
    diff = df["drop_top"] - df["drop_rand"]
    pos = (diff > 0).sum()
    neg = (diff < 0).sum()
    rbc = (pos - neg) / max(1, (pos + neg))

    summary = {
        "dataset": name, "style": style, "k": k, "m": m, "n_peptides": len(df),
        "mean_drop_top": round(df["drop_top"].mean(), 4),
        "mean_drop_rand": round(df["drop_rand"].mean(), 4),
        "mean_drop_bottom": round(df["drop_bottom"].mean(), 4),
        "wilcoxon_p_top_vs_rand": w_p,
        "rank_biserial_top_vs_rand": round(float(rbc), 4),
    }
    return summary, df


# ------------------------------------------------------------------ #
# (b) stability (Spearman across k and across styles)
# ------------------------------------------------------------------ #
def stability_across_k(name, style="X"):
    recs = load_masking(name, style)
    if not recs:
        return None
    ks = CFG.MASK_WINDOWS
    rows = []
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            ka, kb = ks[a], ks[b]
            cors = []
            for r in recs:
                if len(r["per_k"].get(ka, [])) and len(r["per_k"].get(kb, [])):
                    ia = residue_importance_for_k(r, ka)
                    ib = residue_importance_for_k(r, kb)
                    if np.std(ia) > 0 and np.std(ib) > 0:
                        rho, _ = spearmanr(ia, ib)
                        if not np.isnan(rho):
                            cors.append(rho)
            if cors:
                rows.append({"dataset": name, "pair": "k{}_vs_k{}".format(ka, kb),
                             "mean_spearman": round(float(np.mean(cors)), 4),
                             "n": len(cors)})
    return pd.DataFrame(rows)


def stability_across_style(name, s1="X", s2="delete"):
    r1 = load_masking(name, s1)
    r2 = load_masking(name, s2)
    if not r1 or not r2:
        return None
    by_seq = {r["seq"]: r for r in r2}
    cors = []
    for r in r1:
        o = by_seq.get(r["seq"])
        if o is None:
            continue
        a, b = r["residue_importance"], o["residue_importance"]
        if len(a) == len(b) and np.std(a) > 0 and np.std(b) > 0:
            rho, _ = spearmanr(a, b)
            if not np.isnan(rho):
                cors.append(rho)
    if not cors:
        return None
    return {"dataset": name, "pair": "{}_vs_{}".format(s1, s2),
            "mean_spearman": round(float(np.mean(cors)), 4), "n": len(cors)}


# ------------------------------------------------------------------ #
# (c) composition: AA enrichment in top windows vs rest
# ------------------------------------------------------------------ #
def composition(name, style="X", k=5, topq=CFG.MASK_TOPQ):
    recs = load_masking(name, style)
    if not recs:
        return None
    top_counts = {a: 0 for a in AA}
    rest_counts = {a: 0 for a in AA}
    for r in recs:
        deltas = r["per_k"].get(k, [])
        if len(deltas) < 1:
            continue
        order = sorted(deltas, key=lambda x: x[1], reverse=True)
        top_pos = set()
        for st, _ in order[:topq]:
            top_pos.update(range(st, min(st + k, r["L"])))
        for i, aa in enumerate(r["seq"]):
            if aa not in AA:
                continue
            if i in top_pos:
                top_counts[aa] += 1
            else:
                rest_counts[aa] += 1

    t_tot = max(1, sum(top_counts.values()))
    r_tot = max(1, sum(rest_counts.values()))
    rows = []
    for a in AA:
        f_top = top_counts[a] / t_tot
        f_rest = rest_counts[a] / r_tot
        log2e = float(np.log2((f_top + 1e-6) / (f_rest + 1e-6)))
        rows.append({"dataset": name, "aa": a,
                     "freq_top": round(f_top, 4), "freq_rest": round(f_rest, 4),
                     "log2_enrich": round(log2e, 3)})
    return pd.DataFrame(rows).sort_values("log2_enrich", ascending=False)


# ------------------------------------------------------------------ #
# (d) position profile (relative N->C, 10 bins)
# ------------------------------------------------------------------ #
def position_profile(name, style="X", nbins=10):
    recs = load_masking(name, style)
    if not recs:
        return None
    acc = np.zeros(nbins)
    cnt = np.zeros(nbins)
    for r in recs:
        imp = r["residue_importance"]
        L = len(imp)
        if L < 2:
            continue
        for i in range(L):
            b = min(nbins - 1, int(nbins * i / L))
            acc[b] += imp[i]
            cnt[b] += 1
    prof = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return pd.DataFrame({"dataset": name, "bin": range(nbins),
                         "rel_position": [(b + 0.5) / nbins for b in range(nbins)],
                         "mean_importance": np.round(prof, 5)})


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    datasets = sys.argv[1:] or CFG.DATASETS
    del_sum, k_stab, style_stab, comp_all, pos_all, del_raw = [], [], [], [], [], {}

    for name in datasets:
        banner("validate: {}".format(name))
        d = deletion_test(name)
        if d:
            summ, raw = d
            del_sum.append(summ)
            del_raw[name] = raw
            print("  deletion: top={mean_drop_top} rand={mean_drop_rand} "
                  "bottom={mean_drop_bottom}  wilcoxon_p={wilcoxon_p_top_vs_rand:.2e} "
                  "rbc={rank_biserial_top_vs_rand}".format(**summ))
        sk = stability_across_k(name)
        if sk is not None:
            k_stab.append(sk)
            print("  stability across k (mean Spearman):")
            print(sk.to_string(index=False))
        ss = stability_across_style(name)
        if ss:
            style_stab.append(ss)
            print("  stability X vs delete: Spearman={mean_spearman} (n={n})".format(**ss))
        comp = composition(name)
        if comp is not None:
            comp_all.append(comp)
            print("  top enriched AAs:", ", ".join(
                "{}({:+.2f})".format(row.aa, row.log2_enrich) for row in comp.head(5).itertuples()))
        pos = position_profile(name)
        if pos is not None:
            pos_all.append(pos)

    # write everything
    def _w(frames, fn):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                os.path.join(CFG.RESULTS, fn), index=False)
            print("  wrote results/{}".format(fn))

    banner("writing validation CSVs")
    if del_sum:
        pd.DataFrame(del_sum).to_csv(os.path.join(CFG.RESULTS, "deletion_test.csv"), index=False)
        print("  wrote results/deletion_test.csv")
    for name, raw in del_raw.items():
        raw.to_csv(os.path.join(CFG.RESULTS, "deletion_raw_{}.csv".format(name)), index=False)
    _w(k_stab, "stability_across_k.csv")
    if style_stab:
        pd.DataFrame(style_stab).to_csv(os.path.join(CFG.RESULTS, "stability_across_style.csv"), index=False)
        print("  wrote results/stability_across_style.csv")
    _w(comp_all, "composition.csv")
    _w(pos_all, "position_profile.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
