import pytest

from samplics.sampling import SampleSize


size_mean_nat = SampleSize(parameter="mean")


def test_size_nat_wald_basics():
    assert size_mean_nat.parameter == "mean"
    assert size_mean_nat.method == "wald"
    assert not size_mean_nat.stratification


# def test_size_nat_wald_size():
#     size_mean_nat.calculate(0.80, 0.10)
#     assert size_mean_nat.samp_size == 62
#     assert size_mean_nat.deff_c == 1.0
#     assert size_mean_nat.target == 0.80
#     assert size_mean_nat.half_ci == 0.1
