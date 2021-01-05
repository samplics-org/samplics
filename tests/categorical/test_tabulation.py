import pytest
import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator
from samplics.categorical import Tabulation


birthcat = pd.read_csv("./tests/categorical/birthcat.csv")

region = birthcat["region"]
age_cat = birthcat["agecat"]
birth_cat = birthcat["birthcat"]
pop = birthcat["pop"]


tbl_count = Tabulation("count")


@pytest.mark.xfail(strict=True, reason="Parameter not valid")
@pytest.mark.parametrize("param", ["total", "mean", "ratio", "other"])
def test_not_valid_parameter(param):
    tbl = Tabulation(param)


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

tbl_prop = Tabulation("proportion")
tbl_prop.tabulate(birth_cat, remove_nan=True)


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


tbl2_count = Tabulation("count")


def test_oneway_count_two_vars_list():
    tbl2_count.tabulate([region, birth_cat], remove_nan=True)

    assert tbl2_count.table["var_2"][1] == 240
    assert tbl2_count.table["var_2"][2] == 450
    assert tbl2_count.table["var_2"][3] == 233

    assert tbl2_count.table["var_1"][1] == 166
    assert tbl2_count.table["var_1"][2] == 284
    assert tbl2_count.table["var_1"][3] == 250
    assert tbl2_count.table["var_1"][4] == 256


def test_oneway_count_two_vars_tuple():
    tbl2_count.tabulate((region, birth_cat), remove_nan=True)

    assert tbl2_count.table["var_2"][1] == 240
    assert tbl2_count.table["var_2"][2] == 450
    assert tbl2_count.table["var_2"][3] == 233

    assert tbl2_count.table["var_1"][1] == 166
    assert tbl2_count.table["var_1"][2] == 284
    assert tbl2_count.table["var_1"][3] == 250
    assert tbl2_count.table["var_1"][4] == 256


def test_oneway_count_two_vars_pandas():
    tbl2_count.tabulate(birthcat[["region", "birthcat"]], remove_nan=True)

    assert tbl2_count.table["birthcat"][1] == 240
    assert tbl2_count.table["birthcat"][2] == 450
    assert tbl2_count.table["birthcat"][3] == 233

    assert tbl2_count.table["region"][1] == 166
    assert tbl2_count.table["region"][2] == 284
    assert tbl2_count.table["region"][3] == 250
    assert tbl2_count.table["region"][4] == 256


tbl22_count = Tabulation("count")
tbl22_numpy = birthcat[["region", "birthcat"]].to_numpy()
tbl22_count.tabulate(tbl22_numpy, remove_nan=True)


def test_oneway_count_two_vars_numpy():
    assert tbl22_count.table["var_2"][1] == 240
    assert tbl22_count.table["var_2"][2] == 450
    assert tbl22_count.table["var_2"][3] == 233

    assert tbl22_count.table["var_1"][1] == 166
    assert tbl22_count.table["var_1"][2] == 284
    assert tbl22_count.table["var_1"][3] == 250
    assert tbl22_count.table["var_1"][4] == 256


def test_oneway_count_two_vars_stderror():
    assert np.isclose(tbl22_count.stderror["var_2"][1], 13.3337, atol=1e-4)
    assert np.isclose(tbl22_count.stderror["var_2"][2], 15.1940, atol=1e-4)
    assert np.isclose(tbl22_count.stderror["var_2"][3], 13.2050, atol=1e-4)

    assert np.isclose(tbl22_count.stderror["var_1"][1], 11.7183, atol=1e-4)
    assert np.isclose(tbl22_count.stderror["var_1"][2], 14.1365, atol=1e-4)
    assert np.isclose(tbl22_count.stderror["var_1"][3], 13.5947, atol=1e-4)
    assert np.isclose(tbl22_count.stderror["var_1"][4], 13.6983, atol=1e-4)


def test_oneway_count_two_vars_lower_ci():
    assert np.isclose(tbl22_count.lower_ci["var_2"][1], 213.8321, atol=1e-4)
    assert np.isclose(tbl22_count.lower_ci["var_2"][2], 420.1812, atol=1e-4)
    assert np.isclose(tbl22_count.lower_ci["var_2"][3], 207.0847, atol=1e-4)

    assert np.isclose(tbl22_count.lower_ci["var_1"][1], 143.0033, atol=1e-4)
    assert np.isclose(tbl22_count.lower_ci["var_1"][2], 256.2578, atol=1e-4)
    assert np.isclose(tbl22_count.lower_ci["var_1"][3], 223.3210, atol=1e-4)
    assert np.isclose(tbl22_count.lower_ci["var_1"][4], 229.1177, atol=1e-4)


def test_oneway_count_two_vars_upper_ci():
    assert np.isclose(tbl22_count.upper_ci["var_2"][1], 266.1679, atol=1e-4)
    assert np.isclose(tbl22_count.upper_ci["var_2"][2], 479.8188, atol=1e-4)
    assert np.isclose(tbl22_count.upper_ci["var_2"][3], 258.9153, atol=1e-4)

    assert np.isclose(tbl22_count.upper_ci["var_1"][1], 188.9967, atol=1e-4)
    assert np.isclose(tbl22_count.upper_ci["var_1"][2], 311.7422, atol=1e-4)
    assert np.isclose(tbl22_count.upper_ci["var_1"][3], 276.6790, atol=1e-4)
    assert np.isclose(tbl22_count.upper_ci["var_1"][4], 282.8823, atol=1e-4)


tbl3_count = Tabulation("count")
tbl3_numpy = birthcat[["region", "birthcat", "agecat"]].to_numpy()
tbl3_count.tabulate(tbl3_numpy, remove_nan=True)


def test_oneway_count_three_vars_numpy_count():
    assert tbl3_count.table["var_1"][1] == 166
    assert tbl3_count.table["var_1"][2] == 284
    assert tbl3_count.table["var_1"][3] == 250
    assert tbl3_count.table["var_1"][4] == 256

    assert tbl3_count.table["var_2"][1] == 240
    assert tbl3_count.table["var_2"][2] == 450
    assert tbl3_count.table["var_2"][3] == 233

    assert tbl3_count.table["var_3"][1] == 507
    assert tbl3_count.table["var_3"][2] == 316
    assert tbl3_count.table["var_3"][3] == 133


def test_oneway_count_three_vars_numpy_stderror():
    assert np.isclose(tbl3_count.stderror["var_2"][1], 13.3337, atol=1e-4)
    assert np.isclose(tbl3_count.stderror["var_2"][2], 15.1940, atol=1e-4)
    assert np.isclose(tbl3_count.stderror["var_2"][3], 13.2050, atol=1e-4)

    assert np.isclose(tbl3_count.stderror["var_1"][1], 11.7183, atol=1e-4)
    assert np.isclose(tbl3_count.stderror["var_1"][2], 14.1365, atol=1e-4)
    assert np.isclose(tbl3_count.stderror["var_1"][3], 13.5947, atol=1e-4)
    assert np.isclose(tbl3_count.stderror["var_1"][4], 13.6983, atol=1e-4)

    assert np.isclose(tbl3_count.stderror["var_3"][1], 15.4392, atol=1e-4)
    assert np.isclose(tbl3_count.stderror["var_3"][2], 14.5523, atol=1e-4)
    assert np.isclose(tbl3_count.stderror["var_3"][3], 10.7059, atol=1e-4)


tbl2_prop = Tabulation("proportion")
tbl2_pandas = birthcat[["region", "birthcat"]]
tbl2_prop.tabulate(tbl2_pandas, remove_nan=True)


def test_oneway_prop_two_vars_numpy():
    assert np.isclose(tbl2_prop.table["birthcat"][1], 0.2600, atol=1e-4)
    assert np.isclose(tbl2_prop.table["birthcat"][2], 0.4875, atol=1e-4)
    assert np.isclose(tbl2_prop.table["birthcat"][3], 0.2524, atol=1e-4)

    assert np.isclose(tbl2_prop.table["region"][1], 0.1736, atol=1e-4)
    assert np.isclose(tbl2_prop.table["region"][2], 0.2971, atol=1e-4)
    assert np.isclose(tbl2_prop.table["region"][3], 0.2615, atol=1e-4)
    assert np.isclose(tbl2_prop.table["region"][4], 0.2678, atol=1e-4)


def test_oneway_prop_two_vars_stderror():
    assert np.isclose(tbl2_prop.stderror["birthcat"][1], 0.0144, atol=1e-4)
    assert np.isclose(tbl2_prop.stderror["birthcat"][2], 0.0165, atol=1e-4)
    assert np.isclose(tbl2_prop.stderror["birthcat"][3], 0.0143, atol=1e-4)

    assert np.isclose(tbl2_prop.stderror["region"][1], 0.0123, atol=1e-4)
    assert np.isclose(tbl2_prop.stderror["region"][2], 0.0148, atol=1e-4)
    assert np.isclose(tbl2_prop.stderror["region"][3], 0.0142, atol=1e-4)
    assert np.isclose(tbl2_prop.stderror["region"][4], 0.0143, atol=1e-4)


def test_oneway_prop_two_vars_lower_ci():
    assert np.isclose(tbl2_prop.lower_ci["birthcat"][1], 0.2327, atol=1e-4)
    assert np.isclose(tbl2_prop.lower_ci["birthcat"][2], 0.4553, atol=1e-4)
    assert np.isclose(tbl2_prop.lower_ci["birthcat"][3], 0.2254, atol=1e-4)

    assert np.isclose(tbl2_prop.lower_ci["region"][1], 0.1509, atol=1e-4)
    assert np.isclose(tbl2_prop.lower_ci["region"][2], 0.2689, atol=1e-4)
    assert np.isclose(tbl2_prop.lower_ci["region"][3], 0.2346, atol=1e-4)
    assert np.isclose(tbl2_prop.lower_ci["region"][4], 0.2406, atol=1e-4)


def test_oneway_prop_two_vars_upper_ci():
    assert np.isclose(tbl2_prop.upper_ci["birthcat"][1], 0.2894, atol=1e-4)
    assert np.isclose(tbl2_prop.upper_ci["birthcat"][2], 0.5199, atol=1e-4)
    assert np.isclose(tbl2_prop.upper_ci["birthcat"][3], 0.2815, atol=1e-4)

    assert np.isclose(tbl2_prop.upper_ci["region"][1], 0.1990, atol=1e-4)
    assert np.isclose(tbl2_prop.upper_ci["region"][2], 0.3269, atol=1e-4)
    assert np.isclose(tbl2_prop.upper_ci["region"][3], 0.2904, atol=1e-4)
    assert np.isclose(tbl2_prop.upper_ci["region"][4], 0.2968, atol=1e-4)


nhanes = pd.read_csv("./tests/estimation/nhanes.csv")

cholesterol = nhanes["HI_CHOL"]
race = nhanes["race"]
agecat = nhanes["agecat"]
stratum = nhanes["SDMVSTRA"]
psu = nhanes["SDMVPSU"]
weight = nhanes["WTMEC2YR"]

tbl1_nhanes = Tabulation("count")
tbl1_nhanes.tabulate(
    vars=cholesterol, samp_weight=weight, stratum=stratum, psu=psu, remove_nan=True
)


def test_oneway_count_weighted_count():
    assert np.isclose(tbl1_nhanes.table["HI_CHOL"][0], 226710664.8857, atol=1e-4)
    assert np.isclose(tbl1_nhanes.table["HI_CHOL"][1], 28635245.2551, atol=1e-4)


def test_oneway_count_weighted_sdterror():
    assert np.isclose(tbl1_nhanes.stderror["HI_CHOL"][0], 12606884.9914, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stderror["HI_CHOL"][1], 2020710.7438, atol=1e-4)


tbl2_nhanes = Tabulation("proportion")
tbl2_nhanes.tabulate(
    vars=cholesterol, samp_weight=weight, stratum=stratum, psu=psu, remove_nan=True
)


def test_oneway_count_weighted_count():
    assert np.isclose(tbl2_nhanes.table["HI_CHOL"][0], 0.8879, atol=1e-4)
    assert np.isclose(tbl2_nhanes.table["HI_CHOL"][1], 0.1121, atol=1e-4)


def test_oneway_count_weighted_sdterror():
    assert np.isclose(tbl2_nhanes.stderror["HI_CHOL"][0], 0.0054, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stderror["HI_CHOL"][1], 0.0054, atol=1e-4)