import pytest

import numpy as np
import pandas as pd

from samplics.categorical.comparison import Ttest

auto = pd.read_csv("./tests/categorical/auto.csv")

y = auto["mpg"]
make = auto["make"]
foreign = auto["foreign"]


@pytest.mark.xfail(strict=True, reason="Parameters 'known_mean' or 'group' must be provided!")
def test_one_sample_wrong_specifications1():
    one_sample_wrong = Ttest()
    one_sample_wrong.compare(y)


@pytest.mark.xfail(strict=True, reason="Parameters 'known_mean' or 'group' must be provided!")
def test_one_sample_wrong_specifications2():
    one_sample_wrong = Ttest("one-sample")
    one_sample_wrong.compare(y)


@pytest.mark.xfail(
    strict=True,
    reason="Parameter 'type' must be equal to 'one-sample', 'two-sample' or 'many-sample'!",
)
def test_one_sample_wrong_specifications3():
    one_sample_wrong = Ttest("two-sample")
    one_sample_wrong.compare(y, known_mean=0, group=make)


## One-sample with known mean for comparison

one_sample_know_mean = Ttest(type="one-sample")
one_sample_know_mean.compare(y, known_mean=20)


def test_one_sample_known_mean_mean():
    assert np.isclose(one_sample_know_mean.point_est["__none__"], 21.2973, atol=1e-4)


def test_one_sample_known_mean_stderror():
    assert np.isclose(one_sample_know_mean.stderror["__none__"], 0.67255, atol=1e-4)


def test_one_sample_known_mean_stddev():
    assert np.isclose(one_sample_know_mean.stddev["__none__"], 5.78550, atol=1e-4)


def test_one_sample_known_mean_ci():
    assert np.isclose(one_sample_know_mean.lower_ci["__none__"], 19.9569, atol=1e-4)
    assert np.isclose(one_sample_know_mean.upper_ci["__none__"], 22.63769, atol=1e-4)


def test_one_sample_known_mean_stats():
    stats = one_sample_know_mean.stats
    assert np.isclose(stats["number_obs"], 74, atol=1e-4)
    assert np.isclose(stats["t_value"], 1.92889, atol=1e-4)
    assert np.isclose(stats["t_df"], 73, atol=1e-4)
    assert np.isclose(stats["known_mean"], 20, atol=1e-4)
    assert np.isclose(stats["p-value"]["less_than"], 0.9712, atol=1e-4)
    assert np.isclose(stats["p-value"]["greater_than"], 0.0288, atol=1e-4)
    assert np.isclose(stats["p-value"]["not_equal"], 0.0576, atol=1e-4)


# def test_one_sample_known_mean_design_info():
#     assert np.isclose(one_sample_know_mean.design_info["number_strata"], 1, atol=1e-4)
#     assert np.isclose(one_sample_know_mean.design_info["number_psus"], 74, atol=1e-4)
#     assert np.isclose(one_sample_know_mean.design_info["degrees_of_freedom"], 73, atol=1e-4)


## One-sample with comparisons between groups

one_sample_two_group = Ttest(type="one-sample")
one_sample_two_group.compare(y, group=foreign)
# breakpoint()