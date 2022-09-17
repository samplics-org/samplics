import pytest

import numpy as np
import pandas as pd

from samplics.categorical import CrossTabulation
from samplics.estimation import TaylorEstimator


def test_empty_cells():
    df = pd.DataFrame(
        data=[["Woman", "European"]] * 100
        + [["Woman", "American"]] * 35
        + [["Woman", "Other"]] * 93
        + [["Man", "European"]] * 150
        + [["Man", "American"]] * 77,
        columns=["Gender", "Nationality"],
    )
    df["weights"] = [1, 0.3, 8, 3, 0.7] * 91

    crosstab_samplics = CrossTabulation("count")
    crosstab_samplics.tabulate(
        vars=df[["Gender", "Nationality"]],
        samp_weight=df["weights"],
        remove_nan=True,
    )


birthcat = pd.read_csv("./tests/categorical/birthcat.csv")


region = birthcat["region"].to_numpy().astype(int)
age_cat = birthcat["agecat"].to_numpy()
birth_cat = birthcat["birthcat"].to_numpy()
pop = birthcat["pop"]

nhanes = pd.read_csv("./tests/estimation/nhanes.csv")


@pytest.mark.xfail(strict=True, reason="Parameter not valid")
@pytest.mark.parametrize("param", ["total", "mean", "ratio", "other"])
def test_not_valid_parameter(param):
    _ = CrossTabulation(param)


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
    _ = tbl_count.to_dataframe()


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
    assert np.isclose(tbl_count.stats["Pearson-Unadj"]["df"], 4, atol=1e-2)
    assert np.isclose(tbl_count.stats["Pearson-Unadj"]["chisq_value"], 324.2777, atol=1e-4)
    assert np.isclose(tbl_count.stats["Pearson-Unadj"]["p_value"], 0.0000, atol=1e-4)

    assert np.isclose(tbl_count.stats["Pearson-Adj"]["df_num"], 4, atol=1e-2)
    assert np.isclose(tbl_count.stats["Pearson-Adj"]["df_den"], 3688, atol=1e-4)
    assert np.isclose(tbl_count.stats["Pearson-Adj"]["f_value"], 80.9816, atol=1e-4)
    assert np.isclose(tbl_count.stats["Pearson-Adj"]["p_value"], 0.0000, atol=1e-4)


def test_twoway_count_stats_likelihood_ratio():
    assert np.isclose(tbl_count.stats["LR-Unadj"]["df"], 4, atol=1e-2)
    assert np.isclose(tbl_count.stats["LR-Unadj"]["chisq_value"], 302.5142, atol=1e-4)
    assert np.isclose(tbl_count.stats["LR-Unadj"]["p_value"], 0.0000, atol=1e-4)

    assert np.isclose(tbl_count.stats["LR-Adj"]["df_num"], 4, atol=1e-2)
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
    _ = tbl_prop.to_dataframe()


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
    assert np.isclose(tbl_prop.stats["Pearson-Unadj"]["df"], 4, atol=1e-2)
    assert np.isclose(tbl_prop.stats["Pearson-Unadj"]["chisq_value"], 324.2777, atol=1e-4)
    assert np.isclose(tbl_prop.stats["Pearson-Unadj"]["p_value"], 0.0000, atol=1e-4)

    assert np.isclose(tbl_prop.stats["Pearson-Adj"]["df_num"], 4, atol=1e-2)
    assert np.isclose(tbl_prop.stats["Pearson-Adj"]["df_den"], 3688, atol=1e-4)
    assert np.isclose(tbl_prop.stats["Pearson-Adj"]["f_value"], 80.9816, atol=1e-4)
    assert np.isclose(tbl_prop.stats["Pearson-Adj"]["p_value"], 0.0000, atol=1e-4)


def test_twoway_prop_stats_likelihood_ratio():
    assert np.isclose(tbl_prop.stats["LR-Unadj"]["df"], 4, atol=1e-2)
    assert np.isclose(tbl_prop.stats["LR-Unadj"]["chisq_value"], 302.5142, atol=1e-4)
    assert np.isclose(tbl_prop.stats["LR-Unadj"]["p_value"], 0.0000, atol=1e-4)

    assert np.isclose(tbl_prop.stats["LR-Adj"]["df_num"], 4, atol=1e-2)
    assert np.isclose(tbl_prop.stats["LR-Adj"]["df_den"], 3688, atol=1e-4)
    assert np.isclose(tbl_prop.stats["LR-Adj"]["f_value"], 75.5466, atol=1e-4)
    assert np.isclose(tbl_prop.stats["LR-Adj"]["p_value"], 0.0000, atol=1e-4)


def test_twoway_prop_design_info():
    assert tbl_prop.design_info["number_strata"] == 1
    assert tbl_prop.design_info["number_psus"] == 923
    assert tbl_prop.design_info["number_obs"] == 923
    assert tbl_prop.design_info["degrees_of_freedom"] == 922


nhanes = pd.read_csv("./tests/estimation/nhanes.csv")


agecat = nhanes["agecat"]
stratum = nhanes["SDMVSTRA"]
psu = nhanes["SDMVPSU"]
weight = nhanes["WTMEC2YR"]

tbl1_nhanes = CrossTabulation("proportion")
tbl1_nhanes.tabulate(
    vars=nhanes[["race", "HI_CHOL"]], samp_weight=weight, stratum=stratum, psu=psu, remove_nan=True
)


def test_nhanes_twoway_prop_point_est():
    assert np.isclose(tbl1_nhanes.point_est["1"]["0.0"], 0.1368, atol=1e-4)
    assert np.isclose(tbl1_nhanes.point_est["1"]["1.0"], 0.0155, atol=1e-4)
    assert np.isclose(tbl1_nhanes.point_est["2"]["0.0"], 0.5825, atol=1e-4)
    assert np.isclose(tbl1_nhanes.point_est["2"]["1.0"], 0.0807, atol=1e-4)
    assert np.isclose(tbl1_nhanes.point_est["3"]["0.0"], 0.1043, atol=1e-4)
    assert np.isclose(tbl1_nhanes.point_est["3"]["1.0"], 0.0089, atol=1e-4)
    assert np.isclose(tbl1_nhanes.point_est["4"]["0.0"], 0.0642, atol=1e-4)
    assert np.isclose(tbl1_nhanes.point_est["4"]["1.0"], 0.0071, atol=1e-4)


def test_nhanes_twoway_prop_stderror():
    assert np.isclose(tbl1_nhanes.stderror["1"]["0.0"], 0.0270, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stderror["1"]["1.0"], 0.0036, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stderror["2"]["0.0"], 0.0309, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stderror["2"]["1.0"], 0.0059, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stderror["3"]["0.0"], 0.0079, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stderror["3"]["1.0"], 0.0015, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stderror["4"]["0.0"], 0.0098, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stderror["4"]["1.0"], 0.0018, atol=1e-4)


def test_nhanes_twoway_prop_lower_ci():
    assert np.isclose(tbl1_nhanes.lower_ci["1"]["0.0"], 0.0889, atol=1e-4)
    assert np.isclose(tbl1_nhanes.lower_ci["1"]["1.0"], 0.0094, atol=1e-4)
    assert np.isclose(tbl1_nhanes.lower_ci["2"]["0.0"], 0.5159, atol=1e-4)
    assert np.isclose(tbl1_nhanes.lower_ci["2"]["1.0"], 0.0691, atol=1e-4)
    assert np.isclose(tbl1_nhanes.lower_ci["3"]["0.0"], 0.0887, atol=1e-4)
    assert np.isclose(tbl1_nhanes.lower_ci["3"]["1.0"], 0.0062, atol=1e-4)
    assert np.isclose(tbl1_nhanes.lower_ci["4"]["0.0"], 0.0462, atol=1e-4)
    assert np.isclose(tbl1_nhanes.lower_ci["4"]["1.0"], 0.0041, atol=1e-4)


def test_nhanes_twoway_prop_upper_ci():
    assert np.isclose(tbl1_nhanes.upper_ci["1"]["0.0"], 0.2048, atol=1e-4)
    assert np.isclose(tbl1_nhanes.upper_ci["1"]["1.0"], 0.0252, atol=1e-4)
    assert np.isclose(tbl1_nhanes.upper_ci["2"]["0.0"], 0.6462, atol=1e-4)
    assert np.isclose(tbl1_nhanes.upper_ci["2"]["1.0"], 0.0940, atol=1e-4)
    assert np.isclose(tbl1_nhanes.upper_ci["3"]["0.0"], 0.1223, atol=1e-4)
    assert np.isclose(tbl1_nhanes.upper_ci["3"]["1.0"], 0.0128, atol=1e-4)
    assert np.isclose(tbl1_nhanes.upper_ci["4"]["0.0"], 0.0884, atol=1e-4)
    assert np.isclose(tbl1_nhanes.upper_ci["4"]["1.0"], 0.0122, atol=1e-4)


def test_nhanes_twoway_prop_stats_pearson():
    assert np.isclose(tbl1_nhanes.stats["Pearson-Unadj"]["df"], 3, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["Pearson-Unadj"]["chisq_value"], 16.9728, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["Pearson-Unadj"]["p_value"], 0.0007, atol=1e-4)

    assert np.isclose(tbl1_nhanes.stats["Pearson-Adj"]["df_num"], 1.9230, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["Pearson-Adj"]["df_den"], 30.7676, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["Pearson-Adj"]["f_value"], 3.1513, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["Pearson-Adj"]["p_value"], 0.0587, atol=1e-4)


def test_nhanes_twoway_prop_stats_likelihood_ratio():
    assert np.isclose(tbl1_nhanes.stats["LR-Unadj"]["df"], 3, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["LR-Unadj"]["chisq_value"], 17.9643, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["LR-Unadj"]["p_value"], 0.0004, atol=1e-4)

    assert np.isclose(tbl1_nhanes.stats["LR-Adj"]["df_num"], 1.9230, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["LR-Adj"]["df_den"], 30.7676, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["LR-Adj"]["f_value"], 3.3354, atol=1e-4)
    assert np.isclose(tbl1_nhanes.stats["LR-Adj"]["p_value"], 0.0506, atol=1e-4)


def test_nhanes_twoway_prop_design_info():
    assert tbl1_nhanes.design_info["number_strata"] == 15
    assert tbl1_nhanes.design_info["number_psus"] == 31
    assert tbl1_nhanes.design_info["number_obs"] == 7846
    assert tbl1_nhanes.design_info["degrees_of_freedom"] == 16


tbl2_nhanes = CrossTabulation("count")
tbl2_nhanes.tabulate(
    vars=nhanes[["race", "HI_CHOL"]], samp_weight=weight, stratum=stratum, psu=psu, remove_nan=True
)


def test_nhanes_twoway_count_point_est():
    assert np.isclose(tbl2_nhanes.point_est["1"]["0.0"], 34942048.84, atol=1e-2)
    assert np.isclose(tbl2_nhanes.point_est["1"]["1.0"], 3946904.66, atol=1e-2)
    assert np.isclose(tbl2_nhanes.point_est["2"]["0.0"], 148741789.84, atol=1e-2)
    assert np.isclose(tbl2_nhanes.point_est["2"]["1.0"], 20600334.92, atol=1e-2)
    assert np.isclose(tbl2_nhanes.point_est["3"]["0.0"], 26641367.60, atol=1e-2)
    assert np.isclose(tbl2_nhanes.point_est["3"]["1.0"], 2273898.25, atol=1e-2)
    assert np.isclose(tbl2_nhanes.point_est["4"]["0.0"], 16385458.62, atol=1e-2)
    assert np.isclose(tbl2_nhanes.point_est["4"]["1.0"], 1814107.45, atol=1e-2)


def test_nhanes_twoway_count_stderror():
    assert np.isclose(tbl2_nhanes.stderror["1"]["0.0"], 5549735.33, atol=1e-2)
    assert np.isclose(tbl2_nhanes.stderror["1"]["1.0"], 759981.59, atol=1e-2)
    assert np.isclose(tbl2_nhanes.stderror["2"]["0.0"], 15184776.72, atol=1e-2)
    assert np.isclose(tbl2_nhanes.stderror["2"]["1.0"], 2289581.92, atol=1e-2)
    assert np.isclose(tbl2_nhanes.stderror["3"]["0.0"], 2299009.21, atol=1e-2)
    assert np.isclose(tbl2_nhanes.stderror["3"]["1.0"], 384484.38, atol=1e-2)
    assert np.isclose(tbl2_nhanes.stderror["4"]["0.0"], 2497859.99, atol=1e-2)
    assert np.isclose(tbl2_nhanes.stderror["4"]["1.0"], 454779.26, atol=1e-2)


def test_nhanes_twoway_count_lower_ci():
    assert np.isclose(tbl2_nhanes.lower_ci["1"]["0.0"], 23177135.51, atol=1e-2)
    assert np.isclose(tbl2_nhanes.lower_ci["1"]["1.0"], 2335815.66, atol=1e-2)
    assert np.isclose(tbl2_nhanes.lower_ci["2"]["0.0"], 116551501.21, atol=1e-2)
    assert np.isclose(tbl2_nhanes.lower_ci["2"]["1.0"], 15746638.09, atol=1e-2)
    assert np.isclose(tbl2_nhanes.lower_ci["3"]["0.0"], 21767685.79, atol=1e-2)
    assert np.isclose(tbl2_nhanes.lower_ci["3"]["1.0"], 1458827.78, atol=1e-2)
    assert np.isclose(tbl2_nhanes.lower_ci["4"]["0.0"], 11090231.99, atol=1e-2)
    assert np.isclose(tbl2_nhanes.lower_ci["4"]["1.0"], 850018.49, atol=1e-2)


def test_nhanes_twoway_count_upper_ci():
    assert np.isclose(tbl2_nhanes.upper_ci["1"]["0.0"], 46706962.18, atol=1e-2)
    assert np.isclose(tbl2_nhanes.upper_ci["1"]["1.0"], 5557993.67, atol=1e-2)
    assert np.isclose(tbl2_nhanes.upper_ci["2"]["0.0"], 180932078.44, atol=1e-2)
    assert np.isclose(tbl2_nhanes.upper_ci["2"]["1.0"], 25454031.73, atol=1e-2)
    assert np.isclose(tbl2_nhanes.upper_ci["3"]["0.0"], 31515049.42, atol=1e-2)
    assert np.isclose(tbl2_nhanes.upper_ci["3"]["1.0"], 3088968.20, atol=1e-2)
    assert np.isclose(tbl2_nhanes.upper_ci["4"]["0.0"], 21680685.20, atol=1e-2)
    assert np.isclose(tbl2_nhanes.upper_ci["4"]["1.0"], 2778196.39, atol=1e-2)


def test_nhanes_twoway_count_stats_pearson():
    assert np.isclose(tbl2_nhanes.stats["Pearson-Unadj"]["df"], 3, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["Pearson-Unadj"]["chisq_value"], 16.9728, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["Pearson-Unadj"]["p_value"], 0.0007, atol=1e-4)

    assert np.isclose(tbl2_nhanes.stats["Pearson-Adj"]["df_num"], 1.9230, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["Pearson-Adj"]["df_den"], 30.7676, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["Pearson-Adj"]["f_value"], 3.1513, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["Pearson-Adj"]["p_value"], 0.0587, atol=1e-4)


def test_nhanes_twoway_count_stats_likelihood_ratio():
    assert np.isclose(tbl2_nhanes.stats["LR-Unadj"]["df"], 3, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["LR-Unadj"]["chisq_value"], 17.9643, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["LR-Unadj"]["p_value"], 0.0004, atol=1e-4)

    assert np.isclose(tbl2_nhanes.stats["LR-Adj"]["df_num"], 1.9230, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["LR-Adj"]["df_den"], 30.7676, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["LR-Adj"]["f_value"], 3.3354, atol=1e-4)
    assert np.isclose(tbl2_nhanes.stats["LR-Adj"]["p_value"], 0.0506, atol=1e-4)


def test_nhanes_twoway_count_design_info():
    assert tbl2_nhanes.design_info["number_strata"] == 15
    assert tbl2_nhanes.design_info["number_psus"] == 31
    assert tbl2_nhanes.design_info["number_obs"] == 7846
    assert tbl2_nhanes.design_info["degrees_of_freedom"] == 16
