#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/embed.py
==================================================================
Frozen ESM-2 embeddings for the UniDL4BioPep project.

Model: facebook/esm2_t6_8M_UR50D  (6 layers, 320-dim, ~8M params),
loaded through HuggingFace `transformers`, FROZEN (no fine-tuning).

For each peptide we:
  1. tokenize the sequence,
  2. run one forward pass with torch.no_grad(),
  3. take the FINAL-layer residue vectors,
  4. MEAN-POOL them EXCLUDING the start/end special tokens (<cls>/<eos>)
     and any padding, giving ONE 320-dim vector per peptide,
  5. assert the vector is length 320.

Two public functions:
  * embed_sequences(seqs)  -> (N, 320) float32, batched, for any list of
    sequences (also used by the masking code on masked variants).
  * build_dataset_cache(name) -> embeds every row of data/splits/<name>.csv
    once and caches it to data/embeddings/<name>.npz so modelling is fast.

Run:  python -m src.embed            # builds caches for all datasets
      python -m src.embed toxicity   # just one
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.utils import banner

# ------------------------------------------------------------------ #
# lazy global model (loaded once, reused everywhere)
# ------------------------------------------------------------------ #
_TOK = None
_MODEL = None


def _load_model():
    """Load tokenizer + frozen ESM-2 once and cache in module globals."""
    global _TOK, _MODEL
    if _MODEL is not None:
        return _TOK, _MODEL
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(CFG.TORCH_THREADS)
    print("  loading {} (frozen) ...".format(CFG.ESM_MODEL))
    t0 = time.time()
    _TOK = AutoTokenizer.from_pretrained(CFG.ESM_MODEL)
    _MODEL = AutoModel.from_pretrained(CFG.ESM_MODEL)
    _MODEL.eval()                       # inference mode
    for p in _MODEL.parameters():       # freeze: no gradients ever
        p.requires_grad_(False)
    print("  loaded in {:.1f}s".format(time.time() - t0))
    return _TOK, _MODEL


def embed_sequences(seqs, batch_size=CFG.EMB_BATCH, verbose=False):
    """Embed a list of sequences -> (N, 320) float32 numpy array.

    Mean-pools the final hidden layer over REAL residues only (special
    tokens and padding are excluded using the attention mask, with the
    first/last real positions - <cls>/<eos> - also dropped).
    """
    import torch
    tok, model = _load_model()

    # ESM tokenizer wants spaces between residues? No: the ESM tokenizer
    # is character-level and handles a raw sequence string directly.
    vecs = []
    n = len(seqs)
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = list(seqs[start:start + batch_size])
        enc = tok(chunk, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state           # (B, L, 320)
        attn = enc["attention_mask"].clone()     # (B, L): 1 for real+special

        # drop the two special tokens (<cls> at pos0, <eos> at last real pos)
        attn[:, 0] = 0                           # remove <cls>
        lengths = enc["attention_mask"].sum(dim=1)   # real+special length
        for i in range(attn.shape[0]):
            attn[i, lengths[i] - 1] = 0          # remove <eos>

        m = attn.unsqueeze(-1).float()           # (B, L, 1)
        summed = (hidden * m).sum(dim=1)         # (B, 320)
        counts = m.sum(dim=1).clamp(min=1.0)     # (B, 1) -> avoid /0
        pooled = (summed / counts).cpu().numpy().astype(np.float32)
        vecs.append(pooled)
        if verbose and (start // batch_size) % 10 == 0:
            done = min(start + batch_size, n)
            print("    embedded {}/{}  ({:.0f}s)".format(done, n, time.time() - t0))

    arr = np.concatenate(vecs, axis=0)
    assert arr.shape[1] == CFG.EMB_DIM, \
        "expected {}-dim embedding, got {}".format(CFG.EMB_DIM, arr.shape[1])
    return arr


def build_dataset_cache(name, force=False):
    """Embed every row of data/splits/<name>.csv and cache to .npz."""
    import pandas as pd
    out_path = os.path.join(CFG.DATA_EMB, "{}.npz".format(name))
    if os.path.exists(out_path) and not force:
        print("  cache exists, skipping: {}".format(os.path.basename(out_path)))
        return out_path

    df = pd.read_csv(os.path.join(CFG.DATA_SPLIT, "{}.csv".format(name)))
    print("  embedding {} sequences ...".format(len(df)))
    t0 = time.time()
    X = embed_sequences(df["sequence"].tolist(), verbose=True)
    dt = time.time() - t0
    np.savez_compressed(
        out_path,
        X=X,
        y=df["label"].to_numpy().astype(np.int64),
        split=df["split"].to_numpy().astype("U8"),
        sequence=df["sequence"].to_numpy().astype(object),
    )
    print("  wrote {}  shape={}  in {:.1f}s ({:.1f} ms/seq)".format(
        os.path.relpath(out_path, CFG.ROOT), X.shape, dt, 1000 * dt / len(df)))
    return out_path


def load_cache(name):
    """Load a cached embedding file -> dict with X, y, split, sequence."""
    path = os.path.join(CFG.DATA_EMB, "{}.npz".format(name))
    d = np.load(path, allow_pickle=True)
    return {"X": d["X"], "y": d["y"], "split": d["split"], "sequence": d["sequence"]}


def main():
    banner("embed.py - frozen ESM-2 mean-pooled 320-d embeddings")
    targets = sys.argv[1:] or CFG.DATASETS
    for name in targets:
        print("\n>>> {}".format(name))
        build_dataset_cache(name)
    print("\n  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
