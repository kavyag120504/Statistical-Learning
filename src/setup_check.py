#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/setup_check.py
==================================================================
Run this ONCE after cloning the repo on a new machine. It verifies the
demo can run WITHOUT any retraining, and warms the one thing that is not
in the repo: the ESM-2 weights (downloaded from HuggingFace on first use).

What it does:
  1. checks every file the demo needs is present (models, scalers,
     embeddings, metrics),
  2. loads the ESM-2 model once (this triggers the ~30 MB download the
     first time; needs internet ONCE, then it is cached offline),
  3. runs one real end-to-end prediction with a classical model AND the
     CNN, so you know the saved models load and score correctly.

Run:  python -m src.setup_check
Exit code 0 = ready for the demo; non-zero = something is missing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

OK = "[ OK ]"
NO = "[FAIL]"


def _exists(path):
    return os.path.exists(path)


def check_files():
    print("1) Checking required files (no training needed if all present)...")
    missing = []
    for ds in CFG.DATASETS:
        need = [
            os.path.join(CFG.DATA_EMB, "{}.npz".format(ds)),                       # embeddings cache
            os.path.join(CFG.RESULTS, "masking_models", "{}_cnn.pt".format(ds)),   # CNN
            os.path.join(CFG.RESULTS, "masking_models", "{}_scaler.pkl".format(ds)),
        ]
        for m in ["logreg", "svm", "rf", "xgb"]:
            need.append(os.path.join(CFG.ROOT, "models_saved", "{}__{}.joblib".format(ds, m)))
        for p in need:
            tag = OK if _exists(p) else NO
            if not _exists(p):
                missing.append(p)
            print("   {} {}".format(tag, os.path.relpath(p, CFG.ROOT)))
    if missing:
        print("\n   {} {} file(s) missing. If these are the MODELS, run:".format(NO, len(missing)))
        print("       python -m src.save_models")
        print("   If the EMBEDDINGS are missing, run:  python -m src.embed")
        return False
    return True


def check_esm2():
    print("\n2) Loading ESM-2 (downloads ~30 MB on first run; needs internet ONCE)...")
    try:
        from src.embed import embed_sequences
        from src.inference import example_sequence
        seq = example_sequence(CFG.DATASETS[0]) or "ACDEFGHIK"   # real seq from data
        v = embed_sequences([seq])
        print("   {} ESM-2 ready; produced a {}-d embedding.".format(OK, v.shape[1]))
        return True
    except Exception as e:                            # noqa
        print("   {} could not load ESM-2: {}".format(NO, e))
        print("   -> connect to the internet once and re-run; it caches afterwards.")
        return False


def check_predictions():
    print("\n3) Running real end-to-end predictions from the SAVED models...")
    try:
        from src.inference import predict, explain_sequence, label, example_sequence
        ds = CFG.DATASETS[0]
        seq = example_sequence(ds)                    # real active peptide from data
        print("   example peptide ({}): {}".format(ds, seq))
        for m in ["svm", "rf", "xgb", "logreg", "cnn"]:
            p = float(predict(ds, m, seq)[0])
            print("   {} {:<6} P(active)={:.3f} ({})".format(OK, m, p, label(p)))
        imp, p_full = explain_sequence(ds, seq)       # CNN interpretability path
        print("   {} CNN residue-importance explanation works ({} residues).".format(OK, imp.shape[0]))
        return True
    except Exception as e:                            # noqa
        print("   {} prediction failed: {}".format(NO, e))
        return False


def main():
    print("=" * 60)
    print("  UniDL4BioPep demo - setup check")
    print("=" * 60)
    a = check_files()
    b = check_esm2() if a else False
    c = check_predictions() if b else False
    print("\n" + "=" * 60)
    if a and b and c:
        from src.inference import example_sequence
        ds = CFG.DATASETS[0]
        print("  ALL GOOD - the demo is ready. Launch it with:")
        print("      streamlit run app.py")
        print("  or use the command line (example peptide pulled from the data):")
        print("      python -m src.predict -d {} -m cnn --seq {} --explain".format(
            ds, example_sequence(ds)))
        print("=" * 60)
        return 0
    print("  NOT READY - fix the [FAIL] item(s) above, then re-run.")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
