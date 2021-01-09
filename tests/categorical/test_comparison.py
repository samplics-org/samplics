import pytest

import numpy as np
import pandas as pd

from samplics.categorical.comparison import Ttest

auto = pd.read_csv("./tests/categorical/auto2.csv")

y = auto["mpg"]
make = auto["make"]
foreign = auto["foreign"]

y1 = auto["y1"]
y2 = auto["y2"]

# breakpoint()

# np.random.seed(seed=12345)
# y1 = y / 100 + np.random.rand(y.shape[0]) * 1e-5
# y2 = y / 100 + np.random.rand(y.shape[0]) * 1e-5

# auto["y1"] = y1
# auto["y2"] = y2
# auto.to_csv("./tests/categorical/auto2.csv")


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
    assert np.isclose(one_sample_know_mean.point_est, 21.2973, atol=1e-4)


def test_one_sample_known_mean_stderror():
    assert np.isclose(one_sample_know_mean.stderror, 0.67255, atol=1e-4)


def test_one_sample_known_mean_stddev():
    assert np.isclose(one_sample_know_mean.stddev, 5.78550, atol=1e-4)


def test_one_sample_known_mean_ci():
    assert np.isclose(one_sample_know_mean.lower_ci, 19.9569, atol=1e-4)
    assert np.isclose(one_sample_know_mean.upper_ci, 22.63769, atol=1e-4)


def test_one_sample_known_mean_stats():
    stats = one_sample_know_mean.stats
    assert np.isclose(stats["number_obs"], 74, atol=1e-4)
    assert np.isclose(stats["t"], 1.92889, atol=1e-4)
    assert np.isclose(stats["df"], 73, atol=1e-4)
    assert np.isclose(stats["known_mean"], 20, atol=1e-4)
    assert np.isclose(stats["p_value"]["less_than"], 0.9712, atol=1e-4)
    assert np.isclose(stats["p_value"]["greater_than"], 0.0288, atol=1e-4)
    assert np.isclose(stats["p_value"]["not_equal"], 0.0576, atol=1e-4)


## One-sample with comparisons between groups

one_sample_two_groups = Ttest(type="one-sample")
one_sample_two_groups.compare(y, group=foreign)


def test_one_sample_two_groups_means():
    assert np.isclose(one_sample_two_groups.point_est["Domestic"], 19.82692, 1e-4)
    assert np.isclose(one_sample_two_groups.point_est["Foreign"], 24.77273, 1e-4)


def test_one_sample_two_groups_stderror():
    assert np.isclose(one_sample_two_groups.stderror["Domestic"], 0.6558681, 1e-4)
    assert np.isclose(one_sample_two_groups.stderror["Foreign"], 1.3865031, 1e-4)


def test_one_sample_two_groups_stddev():
    assert np.isclose(one_sample_two_groups.stddev["Domestic"], 4.72953, 1e-4)
    assert np.isclose(one_sample_two_groups.stddev["Foreign"], 6.50328, 1e-4)


def test_one_sample_two_groups_lower_ci():
    assert np.isclose(one_sample_two_groups.lower_ci["Domestic"], 18.51978, 1e-4)
    assert np.isclose(one_sample_two_groups.lower_ci["Foreign"], 22.00943, 1e-4)


def test_one_sample_two_groups_upper_ci():
    assert np.isclose(one_sample_two_groups.upper_ci["Domestic"], 21.1340663, 1e-4)
    assert np.isclose(one_sample_two_groups.upper_ci["Foreign"], 27.5360239, 1e-4)


one_sample_stats = one_sample_two_groups.stats


def test_one_sample_two_groups_number_obs():
    assert np.isclose(one_sample_stats["number_obs"]["Domestic"], 52, 1e-4)
    assert np.isclose(one_sample_stats["number_obs"]["Foreign"], 22, 1e-4)


def test_one_sample_two_groups_t_eq_variance():
    assert np.isclose(one_sample_stats["t_eq_variance"], -3.66326, 1e-4)
    assert np.isclose(one_sample_stats["df_eq_variance"], 72, 1e-4)
    assert np.isclose(one_sample_stats["p_value_eq_variance"]["less_than"], 0.0002362, 1e-4)
    assert np.isclose(one_sample_stats["p_value_eq_variance"]["greater_than"], 0.9997638, 1e-4)
    assert np.isclose(one_sample_stats["p_value_eq_variance"]["not_equal"], 0.0004725, 1e-4)


def test_one_sample_two_groups_t_uneq_variance():
    assert np.isclose(one_sample_stats["t_uneq_variance"], -3.22454, 1e-4)
    assert np.isclose(one_sample_stats["df_uneq_variance"], 30.81429, 1e-4)
    assert np.isclose(one_sample_stats["p_value_uneq_variance"]["less_than"], 0.0014909, 1e-4)
    assert np.isclose(one_sample_stats["p_value_uneq_variance"]["greater_than"], 0.9985091, 1e-4)
    assert np.isclose(one_sample_stats["p_value_uneq_variance"]["not_equal"], 0.0029818, 1e-4)


## Two-sample comparisons - UNPAIRED

two_samples_unpaired = Ttest(type="two-sample")
two_samples_unpaired.compare(y, group=foreign)


def test_two_samples_unpaired_means():
    assert np.isclose(two_samples_unpaired.point_est["Domestic"], 19.82692, 1e-4)
    assert np.isclose(two_samples_unpaired.point_est["Foreign"], 24.77273, 1e-4)


def test_two_samples_unpaired_stderror():
    assert np.isclose(two_samples_unpaired.stderror["Domestic"], 0.6577770, 1e-4)
    assert np.isclose(two_samples_unpaired.stderror["Foreign"], 1.4095098, 1e-4)


def test_two_samples_unpaired_stddev():
    assert np.isclose(two_samples_unpaired.stddev["Domestic"], 4.7432972, 1e-4)
    assert np.isclose(two_samples_unpaired.stddev["Foreign"], 6.6111869, 1e-4)


def test_two_samples_unpaired_lower_ci():
    assert np.isclose(two_samples_unpaired.lower_ci["Domestic"], 18.5063807, 1e-4)
    assert np.isclose(two_samples_unpaired.lower_ci["Foreign"], 21.8414912, 1e-4)


def test_two_samples_unpaired_upper_ci():
    assert np.isclose(two_samples_unpaired.upper_ci["Domestic"], 21.1474655, 1e-4)
    assert np.isclose(two_samples_unpaired.upper_ci["Foreign"], 27.7039633, 1e-4)


two_samples_stats= two_samples_unpaired.stats


def test_two_samples_unpaired_number_obs():
    assert np.isclose(two_samples_stats["number_obs"]["Domestic"], 52, 1e-4)
    assert np.isclose(two_samples_stats["number_obs"]["Foreign"], 22, 1e-4)


def test_two_samples_unpaired_t_eq_variance():
    assert np.isclose(two_samples_stats["t_eq_variance"], -3.630848, 1e-4)
    assert np.isclose(two_samples_stats["df_eq_variance"], 72, 1e-4)
    assert np.isclose(two_samples_stats["p_value_eq_variance"]["less_than"], 0.0002627, 1e-4)
    assert np.isclose(two_samples_stats["p_value_eq_variance"]["greater_than"], 0.9997372, 1e-4)
    assert np.isclose(two_samples_stats["p_value_eq_variance"]["not_equal"], 0.0005254, 1e-4)


def test_two_samples_unpaired_t_uneq_variance():
    assert np.isclose(two_samples_stats["t_uneq_variance"], -3.179685, 1e-4)
    assert np.isclose(two_samples_stats["df_uneq_variance"], 30.546278, 1e-4)
    assert np.isclose(two_samples_stats["p_value_uneq_variance"]["less_than"], 0.0016850, 1e-4)
    assert np.isclose(two_samples_stats["p_value_uneq_variance"]["greater_than"], 0.9983150, 1e-4)
    assert np.isclose(two_samples_stats["p_value_uneq_variance"]["not_equal"], 0.0033701, 1e-4)


# breakpoint()

## two-sample with paired observations


# two_sample_paired = Ttest(type="two-sample", paired=True)
# two_sample_paired.compare([y1, y2], group=foreign)


# @pytest.mark.xfail(
#     strict=True, reason="Parameter y must be an array-like object of dimension n by 2!"
# )
# def test_two_sample_wrong_specifications1():
#     two_sample_paired = Ttest(type="two-sample", paired=True)
#     two_sample_paired.compare(y, group=foreign)
