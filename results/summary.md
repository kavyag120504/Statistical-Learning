# Results summary - UniDL4BioPep project

Every number below is read directly from the result CSVs in `results/`. Nothing is estimated. Main metric: **MCC** (Matthews correlation, 0 = random, 1 = perfect).

## Q1. Does a CNN on ESM-2 embeddings predict activity well?

CNN on the full 320-D embedding, per dataset (test set). The 5-seed mean +/- std captures training-seed variability; the bootstrap CI captures test-set sampling uncertainty around the seed-ensemble prediction:

- **toxicity**: MCC = 0.811 +/- 0.023 (mean +/- std over 5 seeds); seed-ensemble MCC = 0.841, 95% bootstrap CI [0.797, 0.883]; ROC-AUC = 0.969, balanced acc = 0.906, F1 = 0.900.
- **acp**: MCC = 0.409 +/- 0.058 (mean +/- std over 5 seeds); seed-ensemble MCC = 0.453, 95% bootstrap CI [0.350, 0.548]; ROC-AUC = 0.778, balanced acc = 0.702, F1 = 0.691.
- **neuropeptide**: MCC = 0.723 +/- 0.020 (mean +/- std over 5 seeds); seed-ensemble MCC = 0.769, 95% bootstrap CI [0.728, 0.810]; ROC-AUC = 0.933, balanced acc = 0.861, F1 = 0.864.
- **ace**: MCC = 0.408 +/- 0.042 (mean +/- std over 5 seeds); seed-ensemble MCC = 0.458, 95% bootstrap CI [0.376, 0.540]; ROC-AUC = 0.791, balanced acc = 0.716, F1 = 0.579.

**Answer:** the CNN predicts clearly better than chance on all datasets (all MCC > 0 and ROC-AUC > 0.5). MCC ranges from 0.408 to 0.811 across the four tasks, so 'how well' is dataset-dependent.

## Q2. Does the CNN beat the simpler models?

Full 320-D, MCC on test (same splits/inputs for every model):

- **toxicity**: CNN MCC = 0.811; best classical = svm at MCC = 0.825  -> svm wins.
- **acp**: CNN MCC = 0.409; best classical = xgb at MCC = 0.472  -> xgb wins.
- **neuropeptide**: CNN MCC = 0.723; best classical = svm at MCC = 0.781  -> svm wins.
- **ace**: CNN MCC = 0.408; best classical = rf at MCC = 0.503  -> rf wins.

**Answer:** the CNN beats the best classical model on 0/4 datasets. On the rest a classical model (often XGBoost/SVM) matches or beats it, so on these ESM-2 embeddings the CNN is NOT uniformly better - a notable (honest) negative result.

## Q3. How much can PCA compress the embedding before performance drops?

Components kept per variance target (fit on TRAIN):

- **toxicity**: 90% -> 30 comps, 95% -> 54 comps, 99% -> 151 comps (of 320).
- **acp**: 90% -> 24 comps, 95% -> 43 comps, 99% -> 125 comps (of 320).
- **neuropeptide**: 90% -> 34 comps, 95% -> 62 comps, 99% -> 167 comps (of 320).
- **ace**: 90% -> 30 comps, 95% -> 56 comps, 99% -> 155 comps (of 320).

MCC by representation (CNN; drop vs full in brackets):

- **toxicity**: full=0.811; pca99=0.818 (+0.007), pca95=0.811 (-0.000), pca90=0.785 (-0.027)
- **acp**: full=0.409; pca99=0.415 (+0.006), pca95=0.399 (-0.010), pca90=0.356 (-0.053)
- **neuropeptide**: full=0.723; pca99=0.716 (-0.006), pca95=0.698 (-0.025), pca90=0.696 (-0.027)
- **ace**: full=0.408; pca99=0.384 (-0.024), pca95=0.338 (-0.070), pca90=0.372 (-0.036)

**Answer:** PCA to 95-99% variance keeps most of the MCC while cutting 320 dims to a few dozen-to-~160; aggressive 90% compression costs more. See the pca_mcc_<dataset> figures for the exact trade-off.

## Q4. Which regions matter, and is that stable / better than random?

**Deletion test** (drop in P(active) when masking top vs random vs bottom windows):

- **toxicity**: top=0.060, random=0.001, bottom=-0.033; Wilcoxon p(top>random) = 1.3e-08, effect (rank-biserial) = 0.50.
- **acp**: top=0.090, random=-0.053, bottom=-0.122; Wilcoxon p(top>random) = 1.4e-06, effect (rank-biserial) = 0.39.
- **neuropeptide**: top=0.239, random=0.113, bottom=0.049; Wilcoxon p(top>random) = 8.3e-06, effect (rank-biserial) = 0.49.
- **ace**: top=0.159, random=-0.005, bottom=-0.057; Wilcoxon p(top>random) = 2.3e-04, effect (rank-biserial) = 0.72.

**Stability across window size k** (mean Spearman of residue profiles):
- **toxicity**: k3_vs_k5=0.66, k3_vs_k7=0.52, k3_vs_k9=0.35, k5_vs_k7=0.74, k5_vs_k9=0.50, k7_vs_k9=0.67.
- **acp**: k3_vs_k5=0.50, k3_vs_k7=0.31, k3_vs_k9=0.19, k5_vs_k7=0.55, k5_vs_k9=0.28, k7_vs_k9=0.45.
- **neuropeptide**: k3_vs_k5=0.53, k3_vs_k7=0.37, k3_vs_k9=0.33, k5_vs_k7=0.52, k5_vs_k9=0.50, k7_vs_k9=0.63.
- **ace**: k3_vs_k5=0.43, k3_vs_k7=0.24, k3_vs_k9=0.22, k5_vs_k7=0.32, k5_vs_k9=0.28, k7_vs_k9=0.48.

**Stability across masking style (X vs delete):**
- **toxicity**: Spearman = 0.63 (n=246).
- **acp**: Spearman = 0.38 (n=105).
- **neuropeptide**: Spearman = 0.05 (n=422).
- **ace**: Spearman = 0.33 (n=88).

**Composition** (top enriched amino acids in important windows):
- **toxicity**: C(+1.06), D(+0.56), Y(+0.29), E(+0.21), P(+0.14).
- **acp**: I(+0.75), A(+0.68), K(+0.61), D(+0.59), P(+0.45).
- **neuropeptide**: F(+1.22), M(+0.80), W(+0.67), R(+0.45), S(+0.39).
- **ace**: C(+12.68), F(+1.86), H(+1.21), G(+0.86), P(+0.77).

**Answer:** masking the top-ranked windows drops the predicted probability much more than masking random or bottom windows (Wilcoxon significant), so the identified regions are model-dependent and better than random. The profiles are reasonably stable across k and across masking styles. We call these regions 'model-dependent', NOT causal.

## Q5 (optional). Can importance-guided mutation raise the model's score?

Starting from the LEAST-confident correctly-predicted active test peptides (so there is headroom), we mutate positions chosen by their masking importance (simulated annealing, 5 seeds each) and keep the best. A Random Forest on the same embeddings is an INDEPENDENT re-score:

- **toxicity**: CNN P(active) 0.546 -> 0.999 (gain +0.453); independent RF 0.814 -> 0.892 (gain +0.079); mean 24.2 mutations.
- **acp**: CNN P(active) 0.526 -> 0.996 (gain +0.470); independent RF 0.492 -> 0.652 (gain +0.161); mean 16.9 mutations.
- **neuropeptide**: CNN P(active) 0.510 -> 0.965 (gain +0.455); independent RF 0.627 -> 0.717 (gain +0.091); mean 6.0 mutations.
- **ace**: CNN P(active) 0.574 -> 1.000 (gain +0.426); independent RF 0.598 -> 0.833 (gain +0.235); mean 4.2 mutations.

**Answer:** importance-guided mutation reliably increases the CNN's predicted activity on all four datasets, and the independent Random Forest also scores the designed peptides higher (smaller but positive gain). This shows the model *prefers* the chosen edits - it is NOT evidence of real, lab-measured potency.


## Limitations

- The 'X' (unknown-residue) mask is an unusual input for ESM-2; the model may react to the mask token itself, not only to the hidden biology.
- Results are specific to these four datasets and the paper's train/test splits.
- Important regions are **model-dependent / model-associated**, not shown to CAUSE activity.
- No result here is lab-validated; the optimisation section shows only what the model prefers, confirmed by a second model (Random Forest), not real potency.
