#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/inference.py
==================================================================
One place that turns a RAW peptide sequence into a prediction, for any of
the five saved models. Both the Streamlit app (app.py) and the CLI
(src/predict.py) use this, so the demo behaves identically everywhere.

What it does for a sequence:
  1. embed it with the frozen ESM-2 (same 320-D mean-pooled vector as
     training),
  2. MIN-MAX scale with the model's TRAIN scaler,
  3. predict P(active).

Models:
  * cnn  -> loaded from results/masking_models/<ds>_cnn.pt (+ scaler),
            via masking.build_predictor (also used for the live
            residue-importance explanation).
  * logreg / svm / rf / xgb -> loaded from models_saved/<ds>__<model>.joblib
            (created by src/save_models.py).

Nothing here retrains anything; it only loads what is already on disk.
"""

import os
import sys
import functools

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.embed import embed_sequences

SAVED_DIR = os.path.join(CFG.ROOT, "models_saved")
MODELS = ["cnn", "logreg", "svm", "rf", "xgb"]

AA = set("ACDEFGHIKLMNPQRSTVWY")


# ------------------------------------------------------------------ #
# validation helper (so the interface gives a friendly error)
# ------------------------------------------------------------------ #
def clean_sequence(seq):
    """Upper-case, strip whitespace; return (clean, problem_or_None)."""
    s = "".join(str(seq).split()).upper()
    if not s:
        return "", "empty sequence"
    bad = sorted({c for c in s if c not in AA})
    if bad:
        return s, "non-standard residue(s): {}".format(", ".join(bad))
    return s, None


# ------------------------------------------------------------------ #
# predictors
# ------------------------------------------------------------------ #
@functools.lru_cache(maxsize=None)
def get_predictor(dataset, model):
    """Return predict(seqs)->np.array of P(active). Cached per (dataset, model)."""
    if dataset not in CFG.DATASETS:
        raise ValueError("unknown dataset {}".format(dataset))
    if model not in MODELS:
        raise ValueError("unknown model {}".format(model))

    if model == "cnn":
        from src.masking import build_predictor
        predict_batch, _ = build_predictor(dataset)      # raw seq -> P(active)
        return predict_batch

    path = os.path.join(SAVED_DIR, "{}__{}.joblib".format(dataset, model))
    if not os.path.exists(path):
        raise FileNotFoundError(
            "{} not found - run `python -m src.save_models {}` first.".format(
                os.path.relpath(path, CFG.ROOT), dataset))
    bundle = joblib.load(path)
    scaler, est = bundle["scaler"], bundle["model"]

    def predict_batch(seqs):
        if isinstance(seqs, str):
            seqs = [seqs]
        emb = embed_sequences(list(seqs))                # (N, 320)
        return est.predict_proba(scaler.transform(emb))[:, 1]

    return predict_batch


def predict(dataset, model, seqs):
    """Convenience: raw sequence(s) -> list of P(active)."""
    if isinstance(seqs, str):
        seqs = [seqs]
    return np.asarray(get_predictor(dataset, model)(list(seqs)))


# ------------------------------------------------------------------ #
# live explanation (CNN only): sliding-window residue importance
# ------------------------------------------------------------------ #
def explain_sequence(dataset, seq, style="X"):
    """Return (residue_importance array, p_full) for one sequence, computed
    live with sliding-window masking. Uses the canonical CNN predictor.

    This is the project's signature interpretability output, produced on the
    fly for whatever sequence the user types in the demo.
    """
    from src.masking import build_predictor, analyse_peptide
    predict_batch, _ = build_predictor(dataset)
    analyse_peptide.style = style
    p_full = float(predict_batch([seq])[0])
    rec = analyse_peptide(seq, predict_batch, p_full, {})
    return rec["residue_importance"], p_full


def label(prob, thr=0.5):
    """Plain-language label for a probability."""
    return "ACTIVE" if prob >= thr else "inactive"


def example_sequence(dataset):
    """A REAL demo peptide for a dataset, taken LIVE from the data - nothing is
    hardcoded. Prefers a confident, correctly-predicted ACTIVE peptide from the
    masking results (so the demo shows a clear positive with real residue
    importance); falls back to an active test sequence straight from the dataset.
    Returns "" only if neither source is available.
    """
    import pickle
    # 1) a correctly-predicted active with non-flat importance, if masking was run
    mpath = os.path.join(CFG.RESULTS, "masking", "{}__X__active.pkl".format(dataset))
    if os.path.exists(mpath):
        try:
            with open(mpath, "rb") as fh:
                recs = pickle.load(fh)
            cand = [r for r in recs if 10 <= r["L"] <= 30 and r["p_full"] >= 0.9
                    and float(np.std(r["residue_importance"])) > 0]
            pool = cand or recs
            if pool:
                return sorted(pool, key=lambda r: -r["p_full"])[0]["seq"]
        except Exception:                      # noqa - fall through to the data
            pass
    # 2) fallback: a real ACTIVE test sequence from the dataset itself
    from src.embed import load_cache
    c = load_cache(dataset)
    mask = (c["y"] == 1)
    if "split" in c:
        mask = mask & (c["split"] == "test")
    seqs = sorted((str(s) for s in c["sequence"][mask]), key=len)
    return seqs[len(seqs) // 2] if seqs else ""   # deterministic median-length active
