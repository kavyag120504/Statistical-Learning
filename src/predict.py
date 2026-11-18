#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/predict.py
==================================================================
Command-line peptide scorer - a no-GUI fallback for the demo (works even
if Streamlit is unavailable on the exam machine).

Example peptides that ARE predicted active (p ~ 1.0) per task - good for a demo:
  toxicity     : CCDWDWCDHICTCCGG
  acp          : FAKKLAKKAKLAKKL
  neuropeptide : QEADPSSSLEADSTLKDEPRELSNM
  ace          : HKEMPFPKYPVQPF

Examples
--------
  # one sequence, default model (cnn) on the toxicity task
  python -m src.predict --dataset toxicity --seq CCDWDWCDHICTCCGG

  # pick a model, add the live residue-importance explanation
  python -m src.predict -d acp -m svm --seq FAKKLAKKAKLAKKL
  python -m src.predict -d toxicity -m cnn --seq CCDWDWCDHICTCCGG --explain

  # score every model at once for comparison
  python -m src.predict -d neuropeptide -m all --seq QEADPSSSLEADSTLKDEPRELSNM
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.inference import predict, explain_sequence, clean_sequence, label, MODELS


def _bar(imp):
    """Tiny text sparkline of residue importance (darker block = more important)."""
    blocks = " .:-=+*#%@"
    if imp.size == 0 or np.allclose(imp, 0):
        return "".join(" " for _ in imp)
    v = np.clip(imp, 0, None)
    v = v / (v.max() + 1e-9)
    return "".join(blocks[min(len(blocks) - 1, int(x * (len(blocks) - 1)))] for x in v)


def main():
    ap = argparse.ArgumentParser(description="Score a peptide with the trained models.")
    ap.add_argument("-d", "--dataset", required=True, choices=CFG.DATASETS)
    ap.add_argument("-m", "--model", default="cnn",
                    choices=MODELS + ["all"])
    ap.add_argument("--seq", required=True, help="peptide sequence (1-letter AAs)")
    ap.add_argument("--explain", action="store_true",
                    help="show live sliding-window residue importance (CNN)")
    args = ap.parse_args()

    seq, problem = clean_sequence(args.seq)
    if problem:
        print("ERROR: {}".format(problem))
        return 2

    models = MODELS if args.model == "all" else [args.model]
    print("\n  dataset : {}".format(args.dataset))
    print("  sequence: {}  (len {})\n".format(seq, len(seq)))
    print("  {:<8} {:>10}   {}".format("model", "P(active)", "call"))
    print("  " + "-" * 34)
    for m in models:
        p = float(predict(args.dataset, m, seq)[0])
        print("  {:<8} {:>10.3f}   {}".format(m, p, label(p)))

    if args.explain:
        imp, p_full = explain_sequence(args.dataset, seq)
        print("\n  residue importance (CNN, style=X; darker = more model-dependent):")
        print("    seq : {}".format(seq))
        print("    imp : {}".format(_bar(imp)))
        order = np.argsort(imp)[::-1][:5]
        tops = ", ".join("{}{}({:.3f})".format(seq[i], i + 1, imp[i]) for i in sorted(order))
        print("    top : {}".format(tops))
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
