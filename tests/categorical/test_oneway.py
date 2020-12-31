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


# tbl_prop = OneWay("proportion")
# tbl_prop.tabulate(birth_cat, remove_nan=True)


# def test_oneway_prop_one_var_count():
#     assert tbl_prop.table["birthcat"][1] == 0.26
#     assert tbl_prop.table["birthcat"][2] == 0.4875
#     assert tbl_prop.table["birthcat"][3] == 0.2524
# exit()


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
    assert np.isclose(tbl_count.stderror["birthcat"][1], 13.3337, atol=1e-4)
    assert np.isclose(tbl_count.stderror["birthcat"][2], 15.1940, atol=1e-4)
    assert np.isclose(tbl_count.stderror["birthcat"][3], 13.2050, atol=1e-4)


def test_oneway_count_one_var_lower_ci():
    assert np.isclose(tbl_count.lower_ci["birthcat"][1], 213.8321, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["birthcat"][2], 420.1812, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["birthcat"][3], 207.0847, atol=1e-4)


def test_oneway_count_one_var_upper_ci():
    assert np.isclose(tbl_count.upper_ci["birthcat"][1], 266.1679, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["birthcat"][2], 479.8188, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["birthcat"][3], 258.9153, atol=1e-4)


def test_oneway_count_one_var_deff_false():
    assert tbl_count.deff["birthcat"] == {}


# tbl_count.tabulate(birth_cat, deff=True, remove_nan=True)


# def test_oneway_count_one_var_deff_true():
#     assert tbl_count.deff["birthcat"][1] == 1
#     assert tbl_count.deff["birthcat"][2] == 1
#     assert tbl_count.deff["birthcat"][3] == 1

tbl_prop = OneWay("proportion")
tbl_prop.tabulate(birth_cat, remove_nan=True)
print(tbl_prop.__dict__)


def test_oneway_prop_one_var_prop():
    assert np.isclose(tbl_prop.table["birthcat"][1], 0.2600, atol=1e-4)
    assert np.isclose(tbl_prop.table["birthcat"][2], 0.4875, atol=1e-4)
    assert np.isclose(tbl_prop.table["birthcat"][3], 0.2524, atol=1e-4)


def test_oneway_prop_one_var_stderr():
    assert np.isclose(tbl_prop.stderror["birthcat"][1], 0.0144, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["birthcat"][2], 0.0165, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["birthcat"][3], 0.0143, atol=1e-4)


def test_oneway_prop_one_var_lower_ci():
    assert np.isclose(tbl_prop.lower_ci["birthcat"][1], 0.2327, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["birthcat"][2], 0.4553, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["birthcat"][3], 0.2254, atol=1e-4)


def test_oneway_prop_one_var_upper_ci():
    assert np.isclose(tbl_prop.upper_ci["birthcat"][1], 0.2894, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["birthcat"][2], 0.5199, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["birthcat"][3], 0.2815, atol=1e-4)


def test_oneway_prop_one_var_deff_false():
    assert tbl_prop.deff["birthcat"] == {}


# def test_two():
#     tbl_count.tabulate([birth_cat, region], remove_nan=True)