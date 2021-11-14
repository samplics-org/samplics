import pytest

import numpy as np

from samplics.sampling import SampleSizeForDifference


size_diff_mean_one_side = SampleSizeForDifference(parameter="mean", two_sides=False)


def test_ss_diff_mean_one_side():
    size_diff_mean_one_side.calculate(delta=2, sigma=3, alpha=0.05, power=0.80)
    assert size_diff_mean_one_side.samp_size == 14
    # assert np.isclose(size_diff_mean_one_side.actual_power == 0.802, 0.001)


size_diff_mean_two_sides = SampleSizeForDifference(parameter="mean")


def test_ss_diff_mean_two_sides():
    size_diff_mean_two_sides.calculate(delta=2, sigma=3, alpha=0.05, power=0.80)
    assert size_diff_mean_two_sides.samp_size == 18
