#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py
==================================================================
Single place for every setting in the UniDL4BioPep project.

Everything that another script might want to change (paths, the ESM-2
model name, the list of random seeds, PCA variance targets, CNN
hyper-parameters, masking window sizes) lives here so the rest of the
code never hard-codes a magic number. Import it like:

    from config import CFG
    print(CFG.SEEDS, CFG.DATASETS)

Plain English on purpose - this file is meant to be read by a human.
"""

import os

# ------------------------------------------------------------------ #
# Paths (all relative to this file, so the project runs from anywhere)
# ------------------------------------------------------------------ #
HERE = os.path.dirname(os.path.abspath(__file__))


class CFG:
    # --- folders --------------------------------------------------- #
    ROOT        = HERE
    DATA_CLEAN  = os.path.join(HERE, "data", "clean")        # input CSVs (done)
    DATA_EMB    = os.path.join(HERE, "data", "embeddings")   # cached ESM-2 vectors
    DATA_SPLIT  = os.path.join(HERE, "data", "splits")       # train/val/test index files
    RESULTS     = os.path.join(HERE, "results")              # CSV / md tables
    FIGURES     = os.path.join(HERE, "figures")              # PDF + PNG plots

    # --- the four tasks (kept SEPARATE, never merged) -------------- #
    # label == 1 means "active / positive" in every one of these files
    # (verified in split.py and stated in the README).
    DATASETS      = ["toxicity", "acp", "neuropeptide", "ace"]
    DEBUG_FIRST   = ["toxicity", "acp"]   # small sets used to debug the pipeline

    # --- reproducibility ------------------------------------------- #
    SEEDS        = [0, 1, 2, 3, 4]        # 5 seeds -> report mean +/- std
    VAL_FRACTION = 0.15                   # carved from TRAIN only

    # --- ESM-2 embedding ------------------------------------------- #
    ESM_MODEL    = "facebook/esm2_t6_8M_UR50D"   # frozen, 320-dim, 6-layer
    EMB_DIM      = 320                            # asserted after pooling
    EMB_BATCH    = 32                             # sequences per forward pass
    TORCH_THREADS = 12                            # CPU threads for torch

    # --- preprocessing / PCA --------------------------------------- #
    # Representations compared in the ablation. "full" = all 320 features.
    PCA_VARIANCE = [0.90, 0.95, 0.99]            # variance targets to keep
    REPRS        = ["full", "pca90", "pca95", "pca99"]

    # --- CNN (UniDL4BioPep replica) -------------------------------- #
    CNN_FILTERS   = 32
    CNN_KERNEL    = 3
    CNN_DROPOUT   = 0.15
    CNN_DENSE     = 64
    CNN_MAX_EPOCH = 200
    CNN_PATIENCE  = 20            # early-stopping patience on val accuracy
    CNN_LR        = 0.05          # SGD start LR (step-decayed)
    CNN_LR_STEP   = 40            # epochs between LR * 0.5 steps
    CNN_LR_GAMMA  = 0.5
    CNN_BATCH     = 64

    # --- masking (the main deliverable) ---------------------------- #
    MASK_WINDOWS  = [3, 5, 7, 9]  # window length k, stride 1
    MASK_STYLES   = ["X", "delete"]  # two styles to check style-robustness
    MASK_TOPQ     = 3             # peptide summary = mean of top-q windows
    DELETION_M    = 3             # #windows masked in the deletion test
    OPT_ROUNDS    = 200           # peptide-optimisation rounds
    OPT_SEEDS     = [0, 1, 2, 3, 4]


# make sure the output folders exist on import (cheap, idempotent)
for _d in (CFG.DATA_EMB, CFG.DATA_SPLIT, CFG.RESULTS, CFG.FIGURES):
    os.makedirs(_d, exist_ok=True)
