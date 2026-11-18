#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/utils.py
==================================================================
Small shared helpers used everywhere: fix all random seeds, log the
library versions (so results are reproducible), and tiny IO helpers.
"""

import os
import sys
import json
import random
import platform

import numpy as np


def set_all_seeds(seed):
    """Fix Python, NumPy and (if present) PyTorch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic CPU math where cheap
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def log_versions(path=None):
    """Collect key library versions. Print them and optionally save to JSON."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "xgboost",
                "torch", "transformers", "matplotlib"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = "NOT INSTALLED"
    print("  library versions:")
    for k, v in info.items():
        print("    {:<12} {}".format(k, v))
    if path:
        with open(path, "w") as fh:
            json.dump(info, fh, indent=2)
    return info


def banner(title):
    """Print a visible section header."""
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)
