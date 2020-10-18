import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator

yrbs = pd.read_csv("./tests/estimation/yrbs.csv")

y = yrbs["qn8"].replace({2: 0})
x = 0.8 * y + 0.5

# print(pd.DataFrame((y,x)))
stratum = yrbs["stratum"]
psu = yrbs["psu"]
weight = yrbs["weight"]


"""Taylor Approximation WITHOUT Stratification for TOTAL"""
svy_total_without_str = TaylorEstimator("total")


def test_total_estimator_without_str():
    total_estimator = svy_total_without_str.estimate(y, weight, psu=psu, remove_nan=True)

    assert np.isclose(total_estimator.point_est["__none__"], 7938.333)
    assert np.isclose(total_estimator.variance["__none__"], 560.0856 ** 2)
    assert np.isclose(total_estimator.stderror["__none__"], 560.0856)
    assert np.isclose(total_estimator.lower_ci["__none__"], 6813.915)
    assert np.isclose(total_estimator.upper_ci["__none__"], 9062.752)
    assert np.isclose(total_estimator.coef_var["__none__"], 560.0856 / 7938.333)


def test_total_estimator_without_str_nor_psu():
    total_estimator = svy_total_without_str.estimate(y, weight, remove_nan=True)

    assert np.isclose(total_estimator.point_est["__none__"], 7938.333)
    assert np.isclose(total_estimator.variance["__none__"], 105.3852 ** 2)
    assert np.isclose(total_estimator.stderror["__none__"], 105.3852)
    assert np.isclose(total_estimator.lower_ci["__none__"], 7731.754)
    assert np.isclose(total_estimator.upper_ci["__none__"], 8144.913)
    assert np.isclose(total_estimator.coef_var["__none__"], 105.3852 / 7938.333)


"""Taylor Approximation WITH Stratification for TOTAL"""
svy_total_with_str = TaylorEstimator("total")


def test_total_estimator_with_str():
    total_estimator = svy_total_with_str.estimate(
        y, weight, stratum=stratum, psu=psu, remove_nan=True
    )

    assert np.isclose(total_estimator.point_est["__none__"], 7938.333)
    assert np.isclose(total_estimator.variance["__none__"], 555.5157 ** 2)
    assert np.isclose(total_estimator.stderror["__none__"], 555.5157)
    assert np.isclose(total_estimator.lower_ci["__none__"], 6814.697)
    assert np.isclose(total_estimator.upper_ci["__none__"], 9061.970)
    assert np.isclose(total_estimator.coef_var["__none__"], 555.5157 / 7938.333)


def test_total_estimator_with_str_without_psu():
    total_estimator = svy_total_with_str.estimate(y, weight, stratum=stratum, remove_nan=True)

    assert np.isclose(total_estimator.point_est["__none__"], 7938.333)
    assert np.isclose(total_estimator.variance["__none__"], 97.75523 ** 2)
    assert np.isclose(total_estimator.stderror["__none__"], 97.75523)
    assert np.isclose(total_estimator.lower_ci["__none__"], 7746.71)
    assert np.isclose(total_estimator.upper_ci["__none__"], 8129.957)
    assert np.isclose(total_estimator.coef_var["__none__"], 97.75523 / 7938.333)


"""Taylor Approximation WITHOUT Stratification for MEAN"""
svy_mean_without_str = TaylorEstimator("mean")


def test_mean_estimator_without_str():
    mean_estimator = svy_mean_without_str.estimate(y, weight, psu=psu, remove_nan=True)

    assert np.isclose(mean_estimator.point_est["__none__"], 0.813_622_5)
    assert np.isclose(mean_estimator.variance["__none__"], 0.020_285_6 ** 2)
    assert np.isclose(mean_estimator.stderror["__none__"], 0.020_285_6)
    assert np.isclose(mean_estimator.lower_ci["__none__"], 0.772_897_5)
    assert np.isclose(mean_estimator.upper_ci["__none__"], 0.854_347_5)
    assert np.isclose(mean_estimator.coef_var["__none__"], 0.020_285_6 / 0.813_622_5)


def test_mean_estimator_without_str_nor_psu():
    mean_estimator = svy_mean_without_str.estimate(y, weight, remove_nan=True)

    assert np.isclose(mean_estimator.point_est["__none__"], 0.8136225)
    assert np.isclose(mean_estimator.variance["__none__"], 0.0066567 ** 2)
    assert np.isclose(mean_estimator.stderror["__none__"], 0.0066567)
    assert np.isclose(mean_estimator.lower_ci["__none__"], 0.8005738)
    assert np.isclose(mean_estimator.upper_ci["__none__"], 0.8266712)
    assert np.isclose(mean_estimator.coef_var["__none__"], 0.0066567 / 0.8136225)


"""Taylor Approximation WITH Stratification for MEAN"""
svy_mean_with_str = TaylorEstimator("mean")


def test_mean_estimator_with_str():
    mean_estimator = svy_mean_with_str.estimate(
        y, weight, stratum=stratum, psu=psu, remove_nan=True
    )

    assert np.isclose(mean_estimator.point_est["__none__"], 0.813_622_5)
    assert np.isclose(mean_estimator.variance["__none__"], 0.019_862_1 ** 2)
    assert np.isclose(mean_estimator.stderror["__none__"], 0.019_862_1)
    assert np.isclose(mean_estimator.lower_ci["__none__"], 0.773_447_6)
    assert np.isclose(mean_estimator.upper_ci["__none__"], 0.853_797_4)
    assert np.isclose(mean_estimator.coef_var["__none__"], 0.019_862_1 / 0.813_622_5)


def test_mean_estimator_with_str_without_psu():
    mean_estimator = svy_mean_with_str.estimate(y, weight, stratum=stratum, remove_nan=True)

    assert np.isclose(mean_estimator.point_est["__none__"], 0.8136225)
    assert np.isclose(mean_estimator.variance["__none__"], 0.0066091 ** 2)
    assert np.isclose(mean_estimator.stderror["__none__"], 0.0066091)
    assert np.isclose(mean_estimator.lower_ci["__none__"], 0.8006671)
    assert np.isclose(mean_estimator.upper_ci["__none__"], 0.8265779)
    assert np.isclose(mean_estimator.coef_var["__none__"], 0.0066091 / 0.8136225)


"""Taylor Approximation WITHOUT Stratification for RATIO"""
svy_ratio_without_str = TaylorEstimator("ratio")


def test_ratio_estimator_without_str():
    ratio_estimator = svy_ratio_without_str.estimate(y, weight, x, psu=psu, remove_nan=True)

    assert np.isclose(ratio_estimator.point_est["__none__"], 0.706_945_8)
    assert np.isclose(ratio_estimator.variance["__none__"], 0.007_657_5 ** 2)
    assert np.isclose(ratio_estimator.stderror["__none__"], 0.007_657_5)
    assert np.isclose(ratio_estimator.lower_ci["__none__"], 0.691_572_8)
    assert np.isclose(ratio_estimator.upper_ci["__none__"], 0.722_318_8)
    assert np.isclose(ratio_estimator.coef_var["__none__"], 0.007_657_5 / 0.706_945_8)


def test_ratio_estimator_without_str_nor_psu():
    ratio_estimator = svy_ratio_without_str.estimate(y, weight, x, remove_nan=True)

    assert np.isclose(ratio_estimator.point_est["__none__"], 0.7069458)
    assert np.isclose(ratio_estimator.variance["__none__"], 0.0025128 ** 2)
    assert np.isclose(ratio_estimator.stderror["__none__"], 0.0025128)
    assert np.isclose(ratio_estimator.lower_ci["__none__"], 0.7020202)
    assert np.isclose(ratio_estimator.upper_ci["__none__"], 0.7118715)
    assert np.isclose(ratio_estimator.coef_var["__none__"], 0.0025128 / 0.7069458)


"""Taylor Approximation WITH Stratification for RATIO"""
svy_ratio_with_str = TaylorEstimator("ratio")


def test_ratio_estimator_with_str():
    ratio_estimator = svy_ratio_with_str.estimate(
        y, weight, x, stratum=stratum, psu=psu, remove_nan=True
    )

    assert np.isclose(ratio_estimator.point_est["__none__"], 0.706_945_8)
    assert np.isclose(ratio_estimator.variance["__none__"], 0.007_497_6 ** 2)
    assert np.isclose(ratio_estimator.stderror["__none__"], 0.007_497_6)
    assert np.isclose(ratio_estimator.lower_ci["__none__"], 0.691_780_5)
    assert np.isclose(ratio_estimator.upper_ci["__none__"], 0.722_111_1)
    assert np.isclose(ratio_estimator.coef_var["__none__"], 0.007_497_6 / 0.706_945_8)


def test_ratio_estimator_with_str_without_psu():
    ratio_estimator = svy_ratio_with_str.estimate(y, weight, x, stratum=stratum, remove_nan=True)

    assert np.isclose(ratio_estimator.point_est["__none__"], 0.7069458)
    assert np.isclose(ratio_estimator.variance["__none__"], 0.0024948 ** 2)
    assert np.isclose(ratio_estimator.stderror["__none__"], 0.0024948)
    assert np.isclose(ratio_estimator.lower_ci["__none__"], 0.7020554)
    assert np.isclose(ratio_estimator.upper_ci["__none__"], 0.7118362)
    assert np.isclose(ratio_estimator.coef_var["__none__"], 0.0024948 / 0.7069458)


"""Taylor Approximation WITHOUT Stratification for PROPORTION"""
svy_prop_without_str = TaylorEstimator("proportion")


def test_prop_estimator_without_str():
    prop_estimator = svy_prop_without_str.estimate(y, weight, psu=psu, remove_nan=True)

    assert np.isclose(prop_estimator.point_est["__none__"][0.0], 0.186_377_5)
    assert np.isclose(prop_estimator.point_est["__none__"][1.0], 0.813_622_5)
    assert np.isclose(prop_estimator.variance["__none__"][0.0], 0.020_285_6 ** 2)
    assert np.isclose(prop_estimator.variance["__none__"][1.0], 0.020_285_6 ** 2)
    assert np.isclose(prop_estimator.stderror["__none__"][0.0], 0.020_285_6)
    assert np.isclose(prop_estimator.stderror["__none__"][1.0], 0.020_285_6)
    assert np.isclose(prop_estimator.lower_ci["__none__"][0.0], 0.149_023)
    assert np.isclose(prop_estimator.lower_ci["__none__"][1.0], 0.769_441_4)
    assert np.isclose(prop_estimator.upper_ci["__none__"][0.0], 0.230_558_6)
    assert np.isclose(prop_estimator.upper_ci["__none__"][1.0], 0.850_977)
    assert np.isclose(prop_estimator.coef_var["__none__"][0.0], 0.020_285_6 / 0.186_377_5)
    assert np.isclose(prop_estimator.coef_var["__none__"][1.0], 0.020_285_6 / 0.813_622_5)


def test_prop_estimator_without_str_nor_psu():
    prop_estimator = svy_prop_without_str.estimate(y, weight, remove_nan=True)

    assert np.isclose(prop_estimator.point_est["__none__"][0.0], 0.1863775)
    assert np.isclose(prop_estimator.point_est["__none__"][1.0], 0.8136225)
    assert np.isclose(prop_estimator.variance["__none__"][0.0], 0.0066567 ** 2)
    assert np.isclose(prop_estimator.variance["__none__"][1.0], 0.0066567 ** 2)
    assert np.isclose(prop_estimator.stderror["__none__"][0.0], 0.0066567)
    assert np.isclose(prop_estimator.stderror["__none__"][1.0], 0.0066567)
    assert np.isclose(prop_estimator.lower_ci["__none__"][0.0], 0.1736793)
    assert np.isclose(prop_estimator.lower_ci["__none__"][1.0], 0.8002204)
    assert np.isclose(prop_estimator.upper_ci["__none__"][0.0], 0.1997796)
    assert np.isclose(prop_estimator.upper_ci["__none__"][1.0], 0.8263207)
    assert np.isclose(prop_estimator.coef_var["__none__"][0.0], 0.0066567 / 0.1863775)
    assert np.isclose(prop_estimator.coef_var["__none__"][1.0], 0.0066567 / 0.8136225)


"""Taylor Approximation WITH Stratification for PROPORTION"""
svy_prop_with_str = TaylorEstimator("proportion")


def test_prop_estimator_with_str():
    prop_estimator = svy_prop_with_str.estimate(
        y, weight, stratum=stratum, psu=psu, remove_nan=True
    )

    assert np.isclose(prop_estimator.point_est["__none__"][0.0], 0.186_377_5)
    assert np.isclose(prop_estimator.point_est["__none__"][1.0], 0.813_622_5)
    assert np.isclose(prop_estimator.variance["__none__"][0.0], 0.019_862_1 ** 2)
    assert np.isclose(prop_estimator.variance["__none__"][1.0], 0.019_862_1 ** 2)
    assert np.isclose(prop_estimator.stderror["__none__"][0.0], 0.019_862_1)
    assert np.isclose(prop_estimator.stderror["__none__"][1.0], 0.019_862_1)
    assert np.isclose(prop_estimator.lower_ci["__none__"][0.0], 0.149_483_7)
    assert np.isclose(prop_estimator.lower_ci["__none__"][1.0], 0.770_084_5)
    assert np.isclose(prop_estimator.upper_ci["__none__"][0.0], 0.229_915_5)
    assert np.isclose(prop_estimator.upper_ci["__none__"][1.0], 0.850_516_3)
    assert np.isclose(prop_estimator.coef_var["__none__"][0.0], 0.019_862_1 / 0.186_377_5)
    assert np.isclose(prop_estimator.coef_var["__none__"][1.0], 0.019_862_1 / 0.813_622_5)


def test_prop_estimator_with_str_without_psu():
    prop_estimator = svy_prop_with_str.estimate(y, weight, stratum=stratum, remove_nan=True)

    assert np.isclose(prop_estimator.point_est["__none__"][0.0], 0.1863775)
    assert np.isclose(prop_estimator.point_est["__none__"][1.0], 0.8136225)
    assert np.isclose(prop_estimator.variance["__none__"][0.0], 0.0066091 ** 2)
    assert np.isclose(prop_estimator.variance["__none__"][1.0], 0.0066091 ** 2)
    assert np.isclose(prop_estimator.stderror["__none__"][0.0], 0.0066091)
    assert np.isclose(prop_estimator.stderror["__none__"][1.0], 0.0066091)
    assert np.isclose(prop_estimator.lower_ci["__none__"][0.0], 0.1737677)
    assert np.isclose(prop_estimator.lower_ci["__none__"][1.0], 0.8003188)
    assert np.isclose(prop_estimator.upper_ci["__none__"][0.0], 0.1996812)
    assert np.isclose(prop_estimator.upper_ci["__none__"][1.0], 0.8262323)
    assert np.isclose(prop_estimator.coef_var["__none__"][0.0], 0.0066091 / 0.1863775)
    assert np.isclose(prop_estimator.coef_var["__none__"][1.0], 0.0066091 / 0.8136225)
