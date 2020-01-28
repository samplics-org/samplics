import math

import numpy as np
import pandas as pd
import pytest
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


def test_total_components_without_str():
    total = svy_total_without_str.get_point(y, weight, exclude_nan=True)
    total_variance = svy_total_without_str.get_variance(y, weight, psu=psu, exclude_nan=True)
    total_lci, total_uci = svy_total_without_str.get_confint(y, weight, psu=psu, exclude_nan=True)
    total_coefvar = svy_total_without_str.get_coefvar(y, weight, psu=psu, exclude_nan=True)

    assert np.isclose(total["__none__"], 7938.333)
    assert np.isclose(total_variance["__none__"], 560.0856 ** 2)
    assert np.isclose(total_lci["__none__"], 6813.915)
    assert np.isclose(total_uci["__none__"], 9062.752)
    assert np.isclose(total_coefvar["__none__"], 560.0856 / 7938.333)


def test_total_estimator_without_str():
    total_estimator = svy_total_without_str.estimate(y, weight, psu=psu, exclude_nan=True)

    assert np.isclose(total_estimator.point_est["__none__"], 7938.333)
    assert np.isclose(total_estimator.variance["__none__"], 560.0856 ** 2)
    assert np.isclose(total_estimator.stderror["__none__"], 560.0856)
    assert np.isclose(total_estimator.lower_ci["__none__"], 6813.915)
    assert np.isclose(total_estimator.upper_ci["__none__"], 9062.752)
    assert np.isclose(total_estimator.coef_var["__none__"], 560.0856 / 7938.333)


"""Taylor Approximation WITH Stratification for TOTAL"""
svy_total_with_str = TaylorEstimator("total")


def test_total_components_with_str():
    total = svy_total_with_str.get_point(y, weight, exclude_nan=True)
    total_variance = svy_total_with_str.get_variance(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )
    total_lci, total_uci = svy_total_with_str.get_confint(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )
    total_coefvar = svy_total_with_str.get_coefvar(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )

    assert np.isclose(total["__none__"], 7938.333)
    assert np.isclose(total_variance["__none__"], 555.5157 ** 2)
    assert np.isclose(total_lci["__none__"], 6814.697)
    assert np.isclose(total_uci["__none__"], 9061.970)


def test_total_estimator_with_str():
    total_estimator = svy_total_with_str.estimate(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )

    assert np.isclose(total_estimator.point_est["__none__"], 7938.333)
    assert np.isclose(total_estimator.variance["__none__"], 555.5157 ** 2)
    assert np.isclose(total_estimator.stderror["__none__"], 555.5157)
    assert np.isclose(total_estimator.lower_ci["__none__"], 6814.697)
    assert np.isclose(total_estimator.upper_ci["__none__"], 9061.970)
    assert np.isclose(total_estimator.coef_var["__none__"], 555.5157 / 7938.333)


"""Taylor Approximation WITHOUT Stratification for MEAN"""
svy_mean_without_str = TaylorEstimator("mean")


def test_mean_components_without_str():
    mean = svy_mean_without_str.get_point(y, weight, exclude_nan=True)
    mean_variance = svy_mean_without_str.get_variance(y, weight, psu=psu, exclude_nan=True)
    mean_lci, mean_uci = svy_mean_without_str.get_confint(y, weight, psu=psu, exclude_nan=True)
    mean_coefvar = svy_mean_without_str.get_coefvar(y, weight, psu=psu, exclude_nan=True)

    assert np.isclose(mean["__none__"], 0.813_622_5)
    assert np.isclose(mean_variance["__none__"], 0.020_285_6 ** 2)
    assert np.isclose(mean_lci["__none__"], 0.772_897_5)
    assert np.isclose(mean_uci["__none__"], 0.854_347_5)
    assert np.isclose(mean_coefvar["__none__"], 0.020_285_6 / 0.813_622_5)


def test_mean_estimator_without_str():
    mean_estimator = svy_mean_without_str.estimate(y, weight, psu=psu, exclude_nan=True)

    assert np.isclose(mean_estimator.point_est["__none__"], 0.813_622_5)
    assert np.isclose(mean_estimator.variance["__none__"], 0.020_285_6 ** 2)
    assert np.isclose(mean_estimator.stderror["__none__"], 0.020_285_6)
    assert np.isclose(mean_estimator.lower_ci["__none__"], 0.772_897_5)
    assert np.isclose(mean_estimator.upper_ci["__none__"], 0.854_347_5)
    assert np.isclose(mean_estimator.coef_var["__none__"], 0.020_285_6 / 0.813_622_5)


"""Taylor Approximation WITH Stratification for MEAN"""
svy_mean_with_str = TaylorEstimator("mean")


def test_mean_components_with_str():
    mean = svy_mean_with_str.get_point(y, weight, exclude_nan=True)
    mean_variance = svy_mean_with_str.get_variance(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )
    mean_lci, mean_uci = svy_mean_with_str.get_confint(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )
    mean_coefvar = svy_mean_with_str.get_coefvar(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )

    assert np.isclose(mean["__none__"], 0.813_622_5)
    assert np.isclose(mean_variance["__none__"], 0.019_862_1 ** 2)
    assert np.isclose(mean_lci["__none__"], 0.773_447_6)
    assert np.isclose(mean_uci["__none__"], 0.853_797_4)


def test_mean_estimator_with_str():
    mean_estimator = svy_mean_with_str.estimate(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )

    assert np.isclose(mean_estimator.point_est["__none__"], 0.813_622_5)
    assert np.isclose(mean_estimator.variance["__none__"], 0.019_862_1 ** 2)
    assert np.isclose(mean_estimator.stderror["__none__"], 0.019_862_1)
    assert np.isclose(mean_estimator.lower_ci["__none__"], 0.773_447_6)
    assert np.isclose(mean_estimator.upper_ci["__none__"], 0.853_797_4)
    assert np.isclose(mean_estimator.coef_var["__none__"], 0.019_862_1 / 0.813_622_5)


"""Taylor Approximation WITHOUT Stratification for RATIO"""
svy_ratio_without_str = TaylorEstimator("ratio")


def test_ratio_components_without_str():
    ratio = svy_ratio_without_str.get_point(y, weight, x, exclude_nan=True)
    ratio_variance = svy_ratio_without_str.get_variance(y, weight, x, psu=psu, exclude_nan=True)
    ratio_lci, ratio_uci = svy_ratio_without_str.get_confint(
        y, weight, x, psu=psu, exclude_nan=True
    )
    ratio_coefvar = svy_ratio_without_str.get_coefvar(y, weight, x, psu=psu, exclude_nan=True)

    assert np.isclose(ratio["__none__"], 0.706_945_8)
    assert np.isclose(ratio_variance["__none__"], 0.007_657_5 ** 2)
    assert np.isclose(ratio_lci["__none__"], 0.691_572_8)
    assert np.isclose(ratio_uci["__none__"], 0.722_318_8)


def test_ratio_estimator_without_str():
    ratio_estimator = svy_ratio_without_str.estimate(y, weight, x, psu=psu, exclude_nan=True)

    assert np.isclose(ratio_estimator.point_est["__none__"], 0.706_945_8)
    assert np.isclose(ratio_estimator.variance["__none__"], 0.007_657_5 ** 2)
    assert np.isclose(ratio_estimator.stderror["__none__"], 0.007_657_5)
    assert np.isclose(ratio_estimator.lower_ci["__none__"], 0.691_572_8)
    assert np.isclose(ratio_estimator.upper_ci["__none__"], 0.722_318_8)
    assert np.isclose(ratio_estimator.coef_var["__none__"], 0.007_657_5 / 0.706_945_8)


"""Taylor Approximation WITH Stratification for RATIO"""
svy_ratio_with_str = TaylorEstimator("ratio")


def test_ratio_components_with_str():
    ratio = svy_ratio_with_str.get_point(y, weight, x, exclude_nan=True)
    ratio_variance = svy_ratio_with_str.get_variance(
        y, weight, x, stratum=stratum, psu=psu, exclude_nan=True
    )
    ratio_lci, ratio_uci = svy_ratio_with_str.get_confint(
        y, weight, x, stratum=stratum, psu=psu, exclude_nan=True
    )
    ratio_coefvar = svy_ratio_with_str.get_coefvar(
        y, weight, x, stratum=stratum, psu=psu, exclude_nan=True
    )

    assert np.isclose(ratio["__none__"], 0.706_945_8)
    assert np.isclose(ratio_variance["__none__"], 0.007_497_6 ** 2)
    assert np.isclose(ratio_lci["__none__"], 0.691_780_5)
    assert np.isclose(ratio_uci["__none__"], 0.722_111_1)


def test_ratio_estimator_with_str():
    ratio_estimator = svy_ratio_with_str.estimate(
        y, weight, x, stratum=stratum, psu=psu, exclude_nan=True
    )

    assert np.isclose(ratio_estimator.point_est["__none__"], 0.706_945_8)
    assert np.isclose(ratio_estimator.variance["__none__"], 0.007_497_6 ** 2)
    assert np.isclose(ratio_estimator.stderror["__none__"], 0.007_497_6)
    assert np.isclose(ratio_estimator.lower_ci["__none__"], 0.691_780_5)
    assert np.isclose(ratio_estimator.upper_ci["__none__"], 0.722_111_1)
    assert np.isclose(ratio_estimator.coef_var["__none__"], 0.007_497_6 / 0.706_945_8)


"""Taylor Approximation WITHOUT Stratification for PROPORTION"""
svy_prop_without_str = TaylorEstimator("proportion")


def test_prop_components_without_str():
    prop = svy_prop_without_str.get_point(y, weight, exclude_nan=True)
    prop_variance = svy_prop_without_str.get_variance(y, weight, psu=psu, exclude_nan=True)
    prop_lci, prop_uci = svy_prop_without_str.get_confint(y, weight, psu=psu, exclude_nan=True)
    prop_coefvar = svy_prop_without_str.get_coefvar(y, weight, psu=psu, exclude_nan=True)

    assert np.isclose(prop["__none__"][0.0], 0.186_377_5)
    assert np.isclose(prop["__none__"][1.0], 0.813_622_5)
    assert np.isclose(prop_variance["__none__"][0.0], 0.020_285_6 ** 2)
    assert np.isclose(prop_variance["__none__"][1.0], 0.020_285_6 ** 2)
    assert np.isclose(prop_lci["__none__"][0.0], 0.149_023)
    assert np.isclose(prop_lci["__none__"][1.0], 0.769_441_4)
    assert np.isclose(prop_uci["__none__"][0.0], 0.230_558_6)
    assert np.isclose(prop_uci["__none__"][1.0], 0.850_977)
    assert np.isclose(prop_coefvar["__none__"][0.0], 0.020_285_6 / 0.186_377_5)
    assert np.isclose(prop_coefvar["__none__"][1.0], 0.020_285_6 / 0.813_622_5)


def test_prop_estimator_without_str():
    prop_estimator = svy_prop_without_str.estimate(y, weight, psu=psu, exclude_nan=True)

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


"""Taylor Approximation WITH Stratification for PROPORTION"""
svy_prop_with_str = TaylorEstimator("proportion")


def test_prop_components_with_str():
    prop = svy_prop_with_str.get_point(y, weight, exclude_nan=True)
    prop_variance = svy_prop_with_str.get_variance(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )
    prop_lci, prop_uci = svy_prop_with_str.get_confint(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )
    prop_coefvar = svy_prop_with_str.get_coefvar(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
    )

    assert np.isclose(prop["__none__"][0.0], 0.186_377_5)
    assert np.isclose(prop["__none__"][1.0], 0.813_622_5)
    assert np.isclose(prop_variance["__none__"][0.0], 0.019_862_1 ** 2)
    assert np.isclose(prop_variance["__none__"][1.0], 0.019_862_1 ** 2)
    assert np.isclose(prop_lci["__none__"][0.0], 0.149_483_7)
    assert np.isclose(prop_lci["__none__"][1.0], 0.770_084_5)
    assert np.isclose(prop_uci["__none__"][0.0], 0.229_915_5)
    assert np.isclose(prop_uci["__none__"][1.0], 0.850_516_3)
    assert np.isclose(prop_coefvar["__none__"][0.0], 0.019_862_1 / 0.186_377_5)
    assert np.isclose(prop_coefvar["__none__"][1.0], 0.019_862_1 / 0.813_622_5)


def test_prop_estimator_with_str():
    prop_estimator = svy_prop_with_str.estimate(
        y, weight, stratum=stratum, psu=psu, exclude_nan=True
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
