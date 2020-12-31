import pytest
import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator
from samplics.categorical import OneWay

birthcat = pd.read_csv("./tests/categorical/birthcat.csv")

region = birthcat["region"]
age_cat = birthcat["agecat"]
birth_cat = birthcat["birthcat"]
pop = birthcat["pop"]


tbl_count = OneWay("count")
# print(tbl_count)


@pytest.mark.xfail(strict=True, reason="Parameter not valid")
@pytest.mark.parametrize("param", ["total", "mean", "ratio", "other"])
def test_not_valid_parameter(param):
    tbl = OneWay(param)


@pytest.mark.xfail(strict=True, reason="Missing values to crash the program")
def test_oneway_nans():
    tbl_count.tabulate(birth_cat)


tbl_count.tabulate(birth_cat, remove_nan=True)


def test_oneway_count_one_var_count():
    assert tbl_count.table["birthcat"][1] == 240
    assert tbl_count.table["birthcat"][2] == 450
    assert tbl_count.table["birthcat"][3] == 233


def test_oneway_count_one_var_stderr():
    assert np.isclose(tbl_count.stderror["birthcat"][1], 13.3337)
    assert np.isclose(tbl_count.stderror["birthcat"][2], 15.1940)
    assert np.isclose(tbl_count.stderror["birthcat"][3], 13.2050)


def test_oneway_count_one_var_lower_ci():
    assert np.isclose(tbl_count.lower_ci["birthcat"][1], 213.8321)
    assert np.isclose(tbl_count.lower_ci["birthcat"][2], 420.1812)
    assert np.isclose(tbl_count.lower_ci["birthcat"][3], 207.0847)


def test_oneway_count_one_var_upper_ci():
    assert np.isclose(tbl_count.upper_ci["birthcat"][1], 266.1679)
    assert np.isclose(tbl_count.upper_ci["birthcat"][2], 479.8188)
    assert np.isclose(tbl_count.upper_ci["birthcat"][3], 258.9153)


def test_oneway_count_one_var_deff_false():
    assert tbl_count.deff["birthcat"] == {}


# tbl_count.tabulate(birth_cat, deff=True, remove_nan=True)


# def test_oneway_count_one_var_deff_true():
#     assert tbl_count.deff["birthcat"][1] == 1
#     assert tbl_count.deff["birthcat"][2] == 1
#     assert tbl_count.deff["birthcat"][3] == 1


# def test_two():
#     tbl_count.tabulate([birth_cat, region], remove_nan=True)