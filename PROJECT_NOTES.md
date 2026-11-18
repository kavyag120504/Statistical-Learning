# Project Notes — UniDL4BioPep (DSC4804 Statistical Learning)

Progress log + handoff doc. Read this first when resuming.

**Project:** Comparative Analysis of Classical and Deep Learning Approaches for
Peptide Bioactivity Classification (base paper: UniDL4BioPep).
**Group:** 18.
**Last updated:** 2026-11-14.

---

## Current status (2026-11-18): full pipeline built; models + metrics DONE.

Done & verified: `src/` package (split, embed, pca_ablation, models, metrics,
masking, validate, optimise, figures, make_summary) + config.py + run_all.py +
README.md + requirements.txt. Embeddings cached (data/embeddings/), **all 400
model cells trained** (results/predictions/), **metrics done**
(results/all_metrics.csv, results/summary_<dataset>.csv).

In progress when paused: **masking** (toxicity X done; rest pending).
Not yet run: validate, optimise, figures, make_summary.

### Resume (tomorrow)
```bash
cd "D:/statistical learning/statistical_learning"
python -m src.masking        # finishes remaining datasets (skips done ones)
python -m src.validate       # deletion test, stability, composition, position
python -m src.optimise       # optional guided mutation + RF cross-check
python -m src.figures        # all plots (PDF + PNG)
python -m src.make_summary   # writes results/summary.md (the 4 answers)
```
(Or `python run_all.py` — cached stages are skipped automatically.)

### Headline results so far (honest)
CNN predicts well (toxicity MCC 0.81, neuropeptide 0.72; acp/ace ~0.41) but is
**never the best model** — SVM/XGB/RF match or beat it on all 4 datasets. PCA to
95–99% variance keeps MCC; 90% costs a little. Full tables in results/.

(Older notes below are from Phase 1.)

---

## What we did (in order)

1. **Identified the task.** Project folder was empty; reconstructed the goal from
   `SL_Project_Charter.docx`, `Project_Charter_UniDL4BioPep.docx`, and
   `UniDL4BioPep_Project_Discussion_v2.pptx` on `D:\`.

2. **Chose 4 datasets** from the UniDL4BioPep repo
   (`https://github.com/dzjxzyd/UniDL4BioPep`), per the project slides:
   - `ace` — ACE-inhibitory activity (primary diagnosis target)
   - `acp` — Anticancer peptides (ACP, main set)
   - `neuropeptide` — Neuropeptides (NeuroPred)
   - `toxicity` — Toxicity (2021 dataset)
   - (`antioxidant` also downloaded as an optional 5th/validation set — not used yet.)

3. **Downloaded raw sources** (train + test `.xlsx`, columns `sequence`,`label`
   where label 1=active/0=inactive) into `data/raw/<dataset>/`.

4. **Assembly step** — `prepare_dataset.py` → `data/processed/`. Raw rows kept
   exactly as-is, just unified into one schema. No cleaning.

5. **Preprocessing step** — `preprocess_dataset.py` → `data/clean/`. Cleaned,
   verified, kept as 4 separate files.

---

## Pipeline / how to reproduce

```bash
cd "D:/statistical learning/statistical_learning"
python prepare_dataset.py       # data/raw/      -> data/processed/  (assemble)
python preprocess_dataset.py    # data/processed -> data/clean/      (clean)
```

Both scripts: print progress, print a validation summary, and **skip missing
sources with a warning instead of crashing**. Re-runnable anytime.
Requires: pandas, openpyxl, numpy (all already installed).

---

## Folder structure

```
statistical_learning/
├─ prepare_dataset.py        # stage 1: assemble raw xlsx -> processed csv
├─ preprocess_dataset.py     # stage 2: clean processed csv -> clean csv
├─ PROJECT_NOTES.md          # this file
└─ data/
   ├─ raw/                   # downloaded source .xlsx (also antioxidant/)
   │   ├─ ace/  acp/  neuropeptide/  toxicity/  antioxidant/
   ├─ processed/             # assembled, raw rows kept as-is
   │   └─ ace.csv acp.csv neuropeptide.csv toxicity.csv
   └─ clean/                 # ✅ USE THESE for modelling
       └─ ace.csv acp.csv neuropeptide.csv toxicity.csv
```

## Schema (all CSVs, same columns)

| column   | meaning                                           |
|----------|---------------------------------------------------|
| dataset  | short id (ace / acp / neuropeptide / toxicity)    |
| sequence | amino-acid sequence (clean = 20 standard AAs, uppercased) |
| length   | sequence length                                   |
| label    | 1 = active/positive, 0 = inactive/negative        |
| split    | "train" or "test" (original paper split preserved)|

---

## Cleaned data summary (`data/clean/`) — what training will see

| dataset      | rows | train | test | pos% | len min/med/max |
|--------------|------|-------|------|------|-----------------|
| ace          | 2352 | 1826  | 526  | 44.8%| 5 / 7 / 81      |
| acp          | 1722 | 1378  | 344  | 50.0%| 2 / 22 / 50     |
| neuropeptide | 4850 | 3880  | 970  | 50.0%| 5 / 14 / 100    |
| toxicity     | 3864 | 3284  | 580  | 50.0%| 13 / 34 / 50    |

Verified across all 4: **0 duplicate sequences, 0 non-standard amino acids,
0 train/test overlap.**

### Cleaning applied (per dataset, in preprocess_dataset.py)
- normalise sequence (strip whitespace, uppercase)
- drop empty sequences
- drop sequences with non-standard residues (keep only `ACDEFGHIKLMNPQRSTVWY`)
- coerce labels to int, drop non-0/1
- drop label conflicts (same sequence with both labels)
- de-duplicate; when a sequence is in both splits, keep the TRAIN copy and
  drop the TEST copy so the test set stays a clean hold-out

Only **ACE** actually needed cleaning: 246 train/test-overlap sequences removed
(its test set went 772 → 526). The other three were already clean in the source.

---

## Key decisions (so we don't re-litigate later)

- **Keep the 4 datasets SEPARATE — do NOT merge.** They are 4 different
  classification tasks; `label=1` means a different thing in each. Train one
  model per dataset and build a comparison table across them (as the paper does).
- **Original train/test split preserved** so results stay comparable to the
  base paper.
- **Two-stage pipeline** (assemble vs clean) kept deliberately separate so the
  raw assembled data in `data/processed/` is always available untouched.

---

## Next steps (not started)

1. **Feature engineering** (for classical baselines) — compute per sequence:
   amino-acid composition, physicochemical descriptors (hydrophobicity, net
   charge, etc.). Output e.g. `data/features/<dataset>.csv`.
2. **Classical ML baselines** — Logistic Regression, SVM, Random Forest,
   XGBoost on the engineered features; evaluate on the test split.
3. **Unsupervised analysis** — PCA (2D viz) + K-means on the feature space.
4. **Deep model** — ESM-2 (`esm2_t6_8M_UR50D`) embeddings + 1D-CNN; Focal Loss
   for any imbalanced set.
5. **PCA on ESM-2 embeddings** — test whether reduced dims retain performance.
6. **Comparison + weak-dataset diagnosis** — Accuracy, F1, MCC, AUC, ROC curves;
   diagnose datasets where the paper underperforms (ACE is the primary target).

Metrics to report throughout: Accuracy, F1, MCC, AUC (+ training time).
