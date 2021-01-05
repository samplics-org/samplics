import pytest
import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator
from samplics.categorical import CrossTabulation


birthcat = pd.read_csv("./tests/categorical/birthcat.csv")

# birthcat.loc[(birthcat["birthcat"] == 2) & (birthcat["region"]==3), "birthcat"] = 3

region = birthcat["region"].to_numpy().astype(int)
age_cat = birthcat["agecat"].to_numpy()
birth_cat = birthcat["birthcat"].to_numpy()
pop = birthcat["pop"]


@pytest.mark.xfail(strict=True, reason="Parameter not valid")
@pytest.mark.parametrize("param", ["total", "mean", "ratio", "other"])
def test_not_valid_parameter(param):
    tbl = CrossTabulation(param)


@pytest.mark.xfail(strict=True, reason="2way tables needs two variables")
def test_twoway_count_one_var_count():
    tbl = CrossTabulation("count")
    tbl.tabulate(region, remove_nan=True)


@pytest.mark.xfail(strict=True, reason="For now, the method will fail if there are missing values")
def test_for_missing_values_in_the_design_matrix():
    tbl_prop = CrossTabulation("proportion")
    tbl_prop.tabulate([region, birth_cat], varnames=["region", "birth_cat"], remove_nan=False)


tbl_count = CrossTabulation("count")
tbl_count.tabulate([age_cat, birth_cat], varnames=["age_cat", "birth_cat"], remove_nan=True)


def test_twoway_count_to_dataframe():
    tbl_df = tbl_count.to_dataframe()


def test_twoway_count_point_est():
    assert tbl_count.point_est["1.0"]["1.0"] == 56
    assert tbl_count.point_est["1.0"]["2.0"] == 239
    assert tbl_count.point_est["1.0"]["3.0"] == 192
    assert tbl_count.point_est["2.0"]["1.0"] == 79
    assert tbl_count.point_est["2.0"]["2.0"] == 193
    assert tbl_count.point_est["2.0"]["3.0"] == 37
    assert tbl_count.point_est["3.0"]["1.0"] == 105
    assert tbl_count.point_est["3.0"]["2.0"] == 18
    assert tbl_count.point_est["3.0"]["3.0"] == 4


def test_twoway_count_sderror():
    assert np.isclose(tbl_count.stderror["1.0"]["1.0"], 7.2567, atol=1e-4)
    assert np.isclose(tbl_count.stderror["1.0"]["2.0"], 13.3156, atol=1e-4)
    assert np.isclose(tbl_count.stderror["1.0"]["3.0"], 12.3380, atol=1e-4)
    assert np.isclose(tbl_count.stderror["2.0"]["1.0"], 8.5039, atol=1e-4)
    assert np.isclose(tbl_count.stderror["2.0"]["2.0"], 12.3616, atol=1e-4)
    assert np.isclose(tbl_count.stderror["2.0"]["3.0"], 5.9628, atol=1e-4)
    assert np.isclose(tbl_count.stderror["3.0"]["1.0"], 9.6517, atol=1e-4)
    assert np.isclose(tbl_count.stderror["3.0"]["2.0"], 4.2033, atol=1e-4)
    assert np.isclose(tbl_count.stderror["3.0"]["3.0"], 1.9967, atol=1e-4)


def test_twoway_count_lower_ci():
    assert np.isclose(tbl_count.lower_ci["1.0"]["1.0"], 41.7585, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["1.0"]["2.0"], 212.8676, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["1.0"]["3.0"], 167.7862, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["2.0"]["1.0"], 62.3107, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["2.0"]["2.0"], 168.7399, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["2.0"]["3.0"], 25.2977, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["3.0"]["1.0"], 86.0581, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["3.0"]["2.0"], 9.7508, atol=1e-4)
    assert np.isclose(tbl_count.lower_ci["3.0"]["3.0"], 0.0813, atol=1e-4)


def test_twoway_count_upper_ci():
    assert np.isclose(tbl_count.upper_ci["1.0"]["1.0"], 70.2415, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["1.0"]["2.0"], 265.1324, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["1.0"]["3.0"], 216.2138, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["2.0"]["1.0"], 95.6893, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["2.0"]["2.0"], 217.2601, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["2.0"]["3.0"], 48.7023, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["3.0"]["1.0"], 123.9419, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["3.0"]["2.0"], 26.2492, atol=1e-4)
    assert np.isclose(tbl_count.upper_ci["3.0"]["3.0"], 7.9187, atol=1e-4)


def test_twoway_count_stats_pearson():
    assert np.isclose(tbl_count.stats["Pearson-Unadj"]["df"], 4, atol=1e-4)
    assert np.isclose(tbl_count.stats["Pearson-Unadj"]["chisq_value"], 324.2777, atol=1e-4)
    assert np.isclose(tbl_count.stats["Pearson-Unadj"]["p_value"], 0.0000, atol=1e-4)

    assert np.isclose(tbl_count.stats["Pearson-Adj"]["df_num"], 4, atol=1e-4)
    assert np.isclose(tbl_count.stats["Pearson-Adj"]["df_den"], 3688, atol=1e-4)
    assert np.isclose(tbl_count.stats["Pearson-Adj"]["f_value"], 80.9816, atol=1e-4)
    assert np.isclose(tbl_count.stats["Pearson-Adj"]["p_value"], 0.0000, atol=1e-4)


def test_twoway_count_stats_likelihood_ratio():
    assert np.isclose(tbl_count.stats["LR-Unadj"]["df"], 4, atol=1e-4)
    assert np.isclose(tbl_count.stats["LR-Unadj"]["chisq_value"], 302.5142, atol=1e-4)
    assert np.isclose(tbl_count.stats["LR-Unadj"]["p_value"], 0.0000, atol=1e-4)

    assert np.isclose(tbl_count.stats["LR-Adj"]["df_num"], 4, atol=1e-4)
    assert np.isclose(tbl_count.stats["LR-Adj"]["df_den"], 3688, atol=1e-4)
    assert np.isclose(tbl_count.stats["LR-Adj"]["f_value"], 75.5466, atol=1e-4)
    assert np.isclose(tbl_count.stats["LR-Adj"]["p_value"], 0.0000, atol=1e-4)


def test_twoway_count_design_info():
    assert tbl_count.design_info["number_strata"] == 1
    assert tbl_count.design_info["number_psus"] == 923
    assert tbl_count.design_info["number_obs"] == 923
    assert tbl_count.design_info["degrees_of_freedom"] == 922


tbl_prop = CrossTabulation("proportion")
tbl_prop.tabulate([age_cat, birth_cat], varnames=["age_cat", "birth_cat"], remove_nan=True)


def test_twoway_prop_to_dataframe():
    tbl_df = tbl_prop.to_dataframe()


def test_twoway_prop_point_est():
    assert np.isclose(tbl_prop.point_est["1.0"]["1.0"], 0.0607, atol=1e-4)
    assert np.isclose(tbl_prop.point_est["1.0"]["2.0"], 0.2589, atol=1e-4)
    assert np.isclose(tbl_prop.point_est["1.0"]["3.0"], 0.2080, atol=1e-4)
    assert np.isclose(tbl_prop.point_est["2.0"]["1.0"], 0.0856, atol=1e-4)
    assert np.isclose(tbl_prop.point_est["2.0"]["2.0"], 0.2091, atol=1e-4)
    assert np.isclose(tbl_prop.point_est["2.0"]["3.0"], 0.0401, atol=1e-4)
    assert np.isclose(tbl_prop.point_est["3.0"]["1.0"], 0.1138, atol=1e-4)
    assert np.isclose(tbl_prop.point_est["3.0"]["2.0"], 0.0195, atol=1e-4)
    assert np.isclose(tbl_prop.point_est["3.0"]["3.0"], 0.0043, atol=1e-4)


def test_twoway_prop_sderror():
    assert np.isclose(tbl_prop.stderror["1.0"]["1.0"], 0.0079, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["1.0"]["2.0"], 0.0144, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["1.0"]["3.0"], 0.0134, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["2.0"]["1.0"], 0.0092, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["2.0"]["2.0"], 0.0134, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["2.0"]["3.0"], 0.0065, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["3.0"]["1.0"], 0.0105, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["3.0"]["2.0"], 0.0046, atol=1e-4)
    assert np.isclose(tbl_prop.stderror["3.0"]["3.0"], 0.0022, atol=1e-4)


def test_twoway_prop_lower_ci():
    assert np.isclose(tbl_prop.lower_ci["1.0"]["1.0"], 0.0470, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["1.0"]["2.0"], 0.2316, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["1.0"]["3.0"], 0.1830, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["2.0"]["1.0"], 0.0692, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["2.0"]["2.0"], 0.1840, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["2.0"]["3.0"], 0.0292, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["3.0"]["1.0"], 0.0948, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["3.0"]["2.0"], 0.0123, atol=1e-4)
    assert np.isclose(tbl_prop.lower_ci["3.0"]["3.0"], 0.0016, atol=1e-4)


def test_twoway_prop_upper_ci():
    assert np.isclose(tbl_prop.upper_ci["1.0"]["1.0"], 0.0781, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["1.0"]["2.0"], 0.2882, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["1.0"]["3.0"], 0.2355, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["2.0"]["1.0"], 0.1055, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["2.0"]["2.0"], 0.2366, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["2.0"]["3.0"], 0.0549, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["3.0"]["1.0"], 0.1359, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["3.0"]["2.0"], 0.0308, atol=1e-4)
    assert np.isclose(tbl_prop.upper_ci["3.0"]["3.0"], 0.0115, atol=1e-4)


def test_twoway_prop_stats_pearson():
    assert np.isclose(tbl_prop.stats["Pearson-Unadj"]["df"], 4, atol=1e-4)
    assert np.isclose(tbl_prop.stats["Pearson-Unadj"]["chisq_value"], 324.2777, atol=1e-4)
    assert np.isclose(tbl_prop.stats["Pearson-Unadj"]["p_value"], 0.0000, atol=1e-4)

    assert np.isclose(tbl_prop.stats["Pearson-Adj"]["df_num"], 4, atol=1e-4)
    assert np.isclose(tbl_prop.stats["Pearson-Adj"]["df_den"], 3688, atol=1e-4)
    assert np.isclose(tbl_prop.stats["Pearson-Adj"]["f_value"], 80.9816, atol=1e-4)
    assert np.isclose(tbl_prop.stats["Pearson-Adj"]["p_value"], 0.0000, atol=1e-4)


def test_twoway_prop_stats_likelihood_ratio():
    assert np.isclose(tbl_prop.stats["LR-Unadj"]["df"], 4, atol=1e-4)
    assert np.isclose(tbl_prop.stats["LR-Unadj"]["chisq_value"], 302.5142, atol=1e-4)
    assert np.isclose(tbl_prop.stats["LR-Unadj"]["p_value"], 0.0000, atol=1e-4)

    assert np.isclose(tbl_prop.stats["LR-Adj"]["df_num"], 4, atol=1e-4)
    assert np.isclose(tbl_prop.stats["LR-Adj"]["df_den"], 3688, atol=1e-4)
    assert np.isclose(tbl_prop.stats["LR-Adj"]["f_value"], 75.5466, atol=1e-4)
    assert np.isclose(tbl_prop.stats["LR-Adj"]["p_value"], 0.0000, atol=1e-4)


def test_twoway_prop_design_info():
    assert tbl_prop.design_info["number_strata"] == 1
    assert tbl_prop.design_info["number_psus"] == 923
    assert tbl_prop.design_info["number_obs"] == 923
    assert tbl_prop.design_info["degrees_of_freedom"] == 922


# nhanes = pd.read_csv("./tests/estimation/nhanes.csv")


# agecat = nhanes["agecat"]
# stratum = nhanes["SDMVSTRA"]
# psu = nhanes["SDMVPSU"]
# weight = nhanes["WTMEC2YR"]

# tbl1_nhanes = CrossTabulation("proportion")
# tbl1_nhanes.tabulate(
#     vars=nhanes[["race", "HI_CHOL"]], samp_weight=weight, stratum=stratum, psu=psu, remove_nan=True
# )
# breakpoint()


# def test_oneway_count_weighted_count():
#     assert np.isclose(tbl1_nhanes.point_est["HI_CHOL"][0], 226710664.8857, atol=1e-4)
#     assert np.isclose(tbl1_nhanes.point_est["HI_CHOL"][1], 28635245.2551, atol=1e-4)


# def test_oneway_count_weighted_sdterror():
#     assert np.isclose(tbl1_nhanes.stderror["HI_CHOL"][0], 12606884.9914, atol=1e-4)
#     assert np.isclose(tbl1_nhanes.stderror["HI_CHOL"][1], 2020710.7438, atol=1e-4)


# tbl2_nhanes = CrossTabulation("proportion")
# tbl2_nhanes.tabulate(
#     vars=nhanes[["race","HI_CHOL"]], samp_weight=weight, stratum=stratum, psu=psu, remove_nan=True
# )


# def test_oneway_count_weighted_count():
#     assert np.isclose(tbl2_nhanes.point_est["HI_CHOL"][0], 0.8879, atol=1e-4)
#     assert np.isclose(tbl2_nhanes.point_est["HI_CHOL"][1], 0.1121, atol=1e-4)


# def test_oneway_count_weighted_sdterror():
#     assert np.isclose(tbl2_nhanes.stderror["HI_CHOL"][0], 0.0054, atol=1e-4)
#     assert np.isclose(tbl2_nhanes.stderror["HI_CHOL"][1], 0.0054, atol=1e-4)