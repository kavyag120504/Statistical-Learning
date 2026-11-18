#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py  -  UniDL4BioPep peptide-activity demo interface
==================================================================
A small Streamlit app for the live demo: type a peptide, pick a task and a
model, get P(active); for the CNN, see the sliding-window residue
importance computed on the fly. Everything loads SAVED models - nothing is
retrained while the teacher watches.

Run (from the project folder `statistical_learning/`):
    streamlit run app.py

If the models are missing, run once first:
    python -m src.save_models
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from src.inference import (predict, explain_sequence, clean_sequence, label,
                           example_sequence, MODELS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f4fd8"


@st.cache_data(show_spinner=False)
def get_example(dataset):
    """Real example peptide pulled from the data at runtime (not hardcoded)."""
    return example_sequence(dataset)


@st.cache_data(show_spinner=False)
def task_caption(dataset):
    """Short data-driven description of a task (counts come from the data)."""
    from src.embed import load_cache
    c = load_cache(dataset)
    n = len(c["y"])
    pos = 100.0 * float((c["y"] == 1).mean())
    return "{} peptides, {:.0f}% active (label 1 vs 0)".format(n, pos)

st.set_page_config(page_title="Peptide activity predictor", page_icon="🧬",
                   layout="wide")


@st.cache_data(show_spinner=False)
def load_metrics():
    p = os.path.join(CFG.RESULTS, "all_metrics.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def importance_figure(seq, imp):
    fig, ax = plt.subplots(figsize=(min(14, 0.32 * len(seq) + 2), 1.7))
    vmax = float(np.abs(imp).max()) or 1.0
    ax.imshow(imp.reshape(1, -1), aspect="auto", cmap="Greys", vmin=0, vmax=vmax)
    ax.set_yticks([])
    ax.set_xticks(range(len(seq)))
    ax.set_xticklabels(list(seq), fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLUE)
        sp.set_linewidth(1.6)
    ax.set_title("Residue importance (darker = model depends on it more)", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# sidebar
# ------------------------------------------------------------------ #
st.sidebar.title("🧬 Settings")
dataset = st.sidebar.selectbox("Task (dataset)", CFG.DATASETS)
st.sidebar.caption(task_caption(dataset))
model = st.sidebar.selectbox("Model", MODELS, index=0)
thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
st.sidebar.caption("Models are pre-trained on CPU and loaded from disk. "
                   "ESM-2 (8M) embeds each sequence in a few milliseconds.")

# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
st.title("Peptide activity predictor")
st.write("Frozen **ESM-2** embedding + trained classifier. "
         "Pick a task on the left, enter a peptide, and predict. "
         "The **CNN** can also explain *which residues* it relied on.")

c1, c2 = st.columns([3, 1])
with c1:
    seq_in = st.text_input("Peptide sequence (1-letter amino acids)",
                           value=get_example(dataset), key="seq_in_" + dataset)
with c2:
    st.write("")
    st.write("")
    use_example = st.button("Use example", use_container_width=True)
if use_example:
    seq_in = get_example(dataset)

seq, problem = clean_sequence(seq_in)

colA, colB = st.columns(2)
go = colA.button("Predict", type="primary", use_container_width=True)
compare = colB.button("Compare all 5 models", use_container_width=True)

if (go or compare) and problem:
    st.error("Invalid sequence - {}".format(problem))
elif go or compare:
    st.caption("Sequence: `{}`  (length {})".format(seq, len(seq)))
    if compare:
        rows = []
        with st.spinner("Scoring with all five models..."):
            for m in MODELS:
                p = float(predict(dataset, m, seq)[0])
                rows.append({"model": m, "P(active)": round(p, 3),
                             "call": label(p, thr)})
        st.subheader("All models on this peptide")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        with st.spinner("Embedding with ESM-2 and predicting..."):
            p = float(predict(dataset, model, seq)[0])
        lab = label(p, thr)
        m1, m2 = st.columns(2)
        m1.metric("P(active)  [{}]".format(model), "{:.3f}".format(p))
        m2.metric("Decision (thr {:.2f})".format(thr), lab)
        st.progress(min(1.0, max(0.0, p)))
        if lab == "ACTIVE":
            st.success("Predicted **ACTIVE** for the {} task.".format(dataset))
        else:
            st.info("Predicted **inactive** for the {} task.".format(dataset))

    # live explanation for the CNN
    if model == "cnn" and not compare:
        with st.spinner("Computing sliding-window residue importance..."):
            imp, p_full = explain_sequence(dataset, seq)
        st.subheader("Why? Residue importance (live sliding-window masking)")
        st.pyplot(importance_figure(seq, imp))
        order = np.argsort(imp)[::-1][:5]
        tops = ", ".join("**{}{}** ({:.3f})".format(seq[i], i + 1, imp[i])
                         for i in sorted(order))
        st.caption("Most model-dependent residues: " + tops)
    elif model != "cnn":
        st.caption("Tip: switch the model to **cnn** to see the residue-importance "
                   "explanation for this peptide.")

# ------------------------------------------------------------------ #
# benchmark context
# ------------------------------------------------------------------ #
with st.expander("Benchmark: how the models compare (test MCC, full 320-D)"):
    allm = load_metrics()
    if allm is None:
        st.write("results/all_metrics.csv not found - run `python -m src.metrics`.")
    else:
        sub = allm[(allm.dataset == dataset) & (allm["repr"] == "full")]
        sub = sub[["model", "mcc_mean", "mcc_std", "roc_auc_mean", "f1_mean"]] \
            .sort_values("mcc_mean", ascending=False).round(3)
        st.dataframe(sub, hide_index=True, use_container_width=True)
        st.caption("MCC = Matthews correlation (0 = random, 1 = perfect), "
                   "mean over 5 seeds. Same splits/inputs for every model.")
