"""Unit tests"""

import random

import numpy as np
import pytest

from anamod.core.compute_p_values import bh_procedure
from anamod import ModelAnalyzer


def test_bh_procedure1():
    """Test BH procedure"""
    pvalues_str = ("0.001 0.008 0.039 0.041 0.042 0.060 0.074 0.205 0.212 0.216 0.222 0.251 0.269 "
                   "0.275 0.34 0.341 0.384 0.569 0.594 0.696 0.762 0.94 0.942 0.975 0.986").split(" ")
    pvalues = ([float(val) for val in pvalues_str])
    pairs = list(zip(range(len(pvalues)), pvalues))
    random.shuffle(pairs)
    idx, pvalues = zip(*pairs)
    adjusted_pvalues, rejected_hypotheses = bh_procedure(pvalues, 0.25)
    zipped = sorted(zip(idx, adjusted_pvalues, rejected_hypotheses), key=lambda elem: elem[0])
    _, adjusted_pvalues, rejected_hypotheses = zip(*zipped)
    assert rejected_hypotheses == tuple([True] * 5 + [False] * 20)
    assert adjusted_pvalues[:5] == (0.001 * 25, 0.008 * 25/2, 0.042 * 25/5, 0.042 * 25/5, 0.042 * 25/5)  # noqa: E226


def test_loss_function_processing():
    """Test loss function processing"""
    targets = np.random.default_rng(0).integers(3, size=100)
    with pytest.raises(ValueError):
        ModelAnalyzer(None, None, targets)
