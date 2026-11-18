#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all.py
==================================================================
One-command pipeline for the whole UniDL4BioPep project. Runs every
stage in order and logs library versions first (for reproducibility).

Stages:
  1. split       - near-dup check + split files
  2. embed       - frozen ESM-2 320-d embeddings (cached)
  3. pca         - PCA variance report
  4. models      - CNN + 4 classical baselines, 5 seeds, 4 representations
  5. metrics     - per-dataset summary tables + bootstrap CIs
  6. masking     - sliding-window importance (both styles)
  7. validate    - deletion test, stability, composition, position
  8. optimise    - OPTIONAL guided mutation + RF cross-check
  9. figures     - all plots (PDF + PNG)

This is slow on CPU (mostly the 5-seed model grid and the masking). It is
fully cached: re-running skips finished embeddings / predictions. Use the
individual `python -m src.<stage>` commands to run one stage at a time.

Run:  python run_all.py
      python run_all.py --skip models   # e.g. if models already ran
"""

import sys
import runpy

from src.utils import log_versions, banner
from config import CFG
import os

STAGES = ["split", "embed", "pca_ablation", "models", "metrics",
          "masking", "validate", "optimise", "figures"]


def main():
    banner("UniDL4BioPep - full pipeline")
    log_versions(os.path.join(CFG.RESULTS, "versions.json"))
    skip = set()
    if "--skip" in sys.argv:
        i = sys.argv.index("--skip")
        skip = set(sys.argv[i + 1:])

    for stage in STAGES:
        if stage in skip or stage.replace("_ablation", "") in skip:
            print("\n[skip] {}".format(stage))
            continue
        banner("STAGE: {}".format(stage))
        # run each stage module as if called with no args
        old = sys.argv
        sys.argv = [stage]
        try:
            runpy.run_module("src." + stage, run_name="__main__")
        except SystemExit:
            pass
        finally:
            sys.argv = old
    banner("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
