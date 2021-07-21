import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator


np.random.seed(12345)

yrbs = pd.read_csv("./tests/estimation/yrbs.csv")

yrbs["y"] = yrbs["qn8"].replace({2: 0})
yrbs["x"] = 0.8 * yrbs["y"] + 0.5
yrbs["domain"] = np.random.choice(["d1", "d2", "d3"], size=yrbs.shape[0], p=[0.1, 0.3, 0.6])
yrbs["by"] = np.random.choice(["b1", "b2"], size=yrbs.shape[0], p=[0.4, 0.6])

# yrbs["fpc"] = 1.0
# yrbs.loc[yrbs["stratum"] == 101, "fpc"] = 0.95
# yrbs.loc[yrbs["stratum"] == 202, "fpc"] = 0.95
# yrbs.loc[yrbs["stratum"] == 214, "fpc"] = 0.95
# yrbs.loc[yrbs["stratum"] == 102, "fpc"] = 0.90
# yrbs.loc[yrbs["stratum"] == 111, "fpc"] = 0.90
# yrbs.loc[yrbs["stratum"] == 213, "fpc"] = 0.90
# yrbs.loc[yrbs["stratum"] == 103, "fpc"] = 0.85
# yrbs.loc[yrbs["stratum"] == 112, "fpc"] = 0.85
# yrbs.loc[yrbs["stratum"] == 212, "fpc"] = 0.85
# yrbs.loc[yrbs["stratum"] == 201, "fpc"] = 0.50
# yrbs.loc[yrbs["stratum"] == 113, "fpc"] = 0.50
# yrbs.to_csv("./tests/estimation/yrbs_modified.csv", index=False)

# print(pd.DataFrame((y,x)))
stratum = yrbs["stratum"]
psu = yrbs["psu"]
weight = yrbs["weight"]
domain = yrbs["domain"]
by = yrbs["by"]
x = yrbs["x"]
y = yrbs["y"]
# fpc_array = yrbs["fpc"]
# fpc_dict = dict(zip(stratum, fpc_array))


"""Taylor Approximation WITHOUT Stratification for TOTAL"""
svy_total_without_str = TaylorEstimator("total")
# svy_total_without_str.estimate(y, weight, psu=psu, by=domain, remove_nan=True)
# breakpoint()


def test_total_estimator_without_str():
    svy_total_without_str.estimate(y, weight, psu=psu, remove_nan=True)

    assert np.isclose(svy_total_without_str.point_est, 7938.333)
    assert np.isclose(svy_total_without_str.variance, 560.0856 ** 2)
    assert np.isclose(svy_total_without_str.stderror, 560.0856)
    assert np.isclose(svy_total_without_str.lower_ci, 6813.915)
    assert np.isclose(svy_total_without_str.upper_ci, 9062.752)
    assert np.isclose(svy_total_without_str.coef_var, 560.0856 / 7938.333)


def test_total_estimator_without_str_nor_psu():
    svy_total_without_str.estimate(y, weight, remove_nan=True)

    assert np.isclose(svy_total_without_str.point_est, 7938.333)
    assert np.isclose(svy_total_without_str.variance, 105.3852 ** 2)
    assert np.isclose(svy_total_without_str.stderror, 105.3852)
    assert np.isclose(svy_total_without_str.lower_ci, 7731.754)
    assert np.isclose(svy_total_without_str.upper_ci, 8144.913)
    assert np.isclose(svy_total_without_str.coef_var, 105.3852 / 7938.333)


svy_total_without_str_domain = TaylorEstimator("total")


def test_total_estimator_without_str_domain():
    svy_total_without_str_domain.estimate(y, weight, psu=psu, domain=domain, remove_nan=True)

    assert np.isclose(svy_total_without_str_domain.point_est["d1"], 759.8535)
    assert np.isclose(svy_total_without_str_domain.point_est["d2"], 2335.8145)
    assert np.isclose(svy_total_without_str_domain.point_est["d3"], 4842.6655)
    assert np.isclose(svy_total_without_str_domain.stderror["d1"], 61.76503)
    assert np.isclose(svy_total_without_str_domain.stderror["d2"], 158.7393)
    assert np.isclose(svy_total_without_str_domain.stderror["d3"], 363.6745)
    assert np.isclose(svy_total_without_str_domain.lower_ci["d1"], 635.855)
    assert np.isclose(svy_total_without_str_domain.lower_ci["d2"], 2017.132)
    assert np.isclose(svy_total_without_str_domain.lower_ci["d3"], 4112.558)
    assert np.isclose(svy_total_without_str_domain.upper_ci["d1"], 883.852)
    assert np.isclose(svy_total_without_str_domain.upper_ci["d2"], 2654.497)
    assert np.isclose(svy_total_without_str_domain.upper_ci["d3"], 5572.773)


def test_total_estimator_without_str_nor_psu_domain():
    svy_total_without_str_domain.estimate(y, weight, domain=domain, remove_nan=True)

    assert np.isclose(svy_total_without_str_domain.point_est["d1"], 759.8535)
    assert np.isclose(svy_total_without_str_domain.point_est["d2"], 2335.8145)
    assert np.isclose(svy_total_without_str_domain.point_est["d3"], 4842.6655)
    assert np.isclose(svy_total_without_str_domain.stderror["d1"], 40.45371)
    assert np.isclose(svy_total_without_str_domain.stderror["d2"], 67.67121)
    assert np.isclose(svy_total_without_str_domain.stderror["d3"], 93.37974)
    assert np.isclose(svy_total_without_str_domain.lower_ci["d1"], 680.5547)
    assert np.isclose(svy_total_without_str_domain.lower_ci["d2"], 2203.163)
    assert np.isclose(svy_total_without_str_domain.lower_ci["d3"], 4659.619)
    assert np.isclose(svy_total_without_str_domain.upper_ci["d1"], 839.1523)
    assert np.isclose(svy_total_without_str_domain.upper_ci["d2"], 2468.466)
    assert np.isclose(svy_total_without_str_domain.upper_ci["d3"], 5025.712)


"""Taylor Approximation WITH Stratification for TOTAL"""
svy_total_with_str = TaylorEstimator("total")


def test_total_estimator_with_str():
    svy_total_with_str.estimate(y, weight, stratum=stratum, psu=psu, remove_nan=True)

    assert np.isclose(svy_total_with_str.point_est, 7938.333)
    assert np.isclose(svy_total_with_str.variance, 555.5157 ** 2)
    assert np.isclose(svy_total_with_str.stderror, 555.5157)
    assert np.isclose(svy_total_with_str.lower_ci, 6814.697)
    assert np.isclose(svy_total_with_str.upper_ci, 9061.970)
    assert np.isclose(svy_total_with_str.coef_var, 555.5157 / 7938.333)


def test_total_estimator_with_str_without_psu():
    svy_total_with_str.estimate(y, weight, stratum=stratum, remove_nan=True)

    assert np.isclose(svy_total_with_str.point_est, 7938.333)
    assert np.isclose(svy_total_with_str.variance, 97.75523 ** 2)
    assert np.isclose(svy_total_with_str.stderror, 97.75523)
    assert np.isclose(svy_total_with_str.lower_ci, 7746.71)
    assert np.isclose(svy_total_with_str.upper_ci, 8129.957)
    assert np.isclose(svy_total_with_str.coef_var, 97.75523 / 7938.333)


svy_total_with_str_domain = TaylorEstimator("total")


def test_total_estimator_with_str_domain():
    svy_total_with_str_domain.estimate(
        y, weight, stratum=stratum, psu=psu, domain=domain, remove_nan=True
    )

    assert np.isclose(svy_total_with_str_domain.point_est["d1"], 759.8535)
    assert np.isclose(svy_total_with_str_domain.point_est["d2"], 2335.8145)
    assert np.isclose(svy_total_with_str_domain.point_est["d3"], 4842.6655)
    assert np.isclose(svy_total_with_str_domain.stderror["d1"], 60.89981)
    assert np.isclose(svy_total_with_str_domain.stderror["d2"], 159.748)
    assert np.isclose(svy_total_with_str_domain.stderror["d3"], 360.2601)
    assert np.isclose(svy_total_with_str_domain.lower_ci["d1"], 636.672)
    assert np.isclose(svy_total_with_str_domain.lower_ci["d2"], 2012.694)
    assert np.isclose(svy_total_with_str_domain.lower_ci["d3"], 4113.971)
    assert np.isclose(svy_total_with_str_domain.upper_ci["d1"], 883.035)
    assert np.isclose(svy_total_with_str_domain.upper_ci["d2"], 2658.935)
    assert np.isclose(svy_total_with_str_domain.upper_ci["d3"], 5571.36)


def test_total_estimator_with_str_without_psu_domain():
    svy_total_with_str_domain.estimate(y, weight, stratum=stratum, domain=domain, remove_nan=True)

    assert np.isclose(svy_total_with_str_domain.point_est["d1"], 759.8535)
    assert np.isclose(svy_total_with_str_domain.point_est["d2"], 2335.8145)
    assert np.isclose(svy_total_with_str_domain.point_est["d3"], 4842.6655)
    assert np.isclose(svy_total_with_str_domain.stderror["d1"], 40.31828)
    assert np.isclose(svy_total_with_str_domain.stderror["d2"], 66.78249)
    assert np.isclose(svy_total_with_str_domain.stderror["d3"], 90.01087)
    assert np.isclose(svy_total_with_str_domain.lower_ci["d1"], 680.8202)
    assert np.isclose(svy_total_with_str_domain.lower_ci["d2"], 2204.905)
    assert np.isclose(svy_total_with_str_domain.lower_ci["d3"], 4666.223)
    assert np.isclose(svy_total_with_str_domain.upper_ci["d1"], 838.8868)
    assert np.isclose(svy_total_with_str_domain.upper_ci["d2"], 2466.724)
    assert np.isclose(svy_total_with_str_domain.upper_ci["d3"], 5019.108)


"""Taylor Approximation WITHOUT Stratification for MEAN"""
svy_mean_without_str = TaylorEstimator("mean")


def test_mean_estimator_without_str():
    svy_mean_without_str.estimate(y, weight, psu=psu, remove_nan=True)

    assert np.isclose(svy_mean_without_str.point_est, 0.813_622_5)
    assert np.isclose(svy_mean_without_str.variance, 0.020_285_6 ** 2)
    assert np.isclose(svy_mean_without_str.stderror, 0.020_285_6)
    assert np.isclose(svy_mean_without_str.lower_ci, 0.772_897_5)
    assert np.isclose(svy_mean_without_str.upper_ci, 0.854_347_5)
    assert np.isclose(svy_mean_without_str.coef_var, 0.020_285_6 / 0.813_622_5)


def test_mean_estimator_without_str_nor_psu():
    svy_mean_without_str.estimate(y, weight, remove_nan=True)

    assert np.isclose(svy_mean_without_str.point_est, 0.8136225)
    assert np.isclose(svy_mean_without_str.variance, 0.0066567 ** 2)
    assert np.isclose(svy_mean_without_str.stderror, 0.0066567)
    assert np.isclose(svy_mean_without_str.lower_ci, 0.8005738)
    assert np.isclose(svy_mean_without_str.upper_ci, 0.8266712)
    assert np.isclose(svy_mean_without_str.coef_var, 0.0066567 / 0.8136225)


svy_mean_without_str_domain = TaylorEstimator("mean")


def test_mean_estimator_without_str_domain():
    svy_mean_without_str_domain.estimate(y, weight, psu=psu, domain=domain, remove_nan=True)

    assert np.isclose(svy_mean_without_str_domain.point_est["d1"], 0.8311598)
    assert np.isclose(svy_mean_without_str_domain.point_est["d2"], 0.797226)
    assert np.isclose(svy_mean_without_str_domain.point_est["d3"], 0.8190359)
    assert np.isclose(svy_mean_without_str_domain.stderror["d1"], 0.0203778)
    assert np.isclose(svy_mean_without_str_domain.stderror["d2"], 0.0260659)
    assert np.isclose(svy_mean_without_str_domain.stderror["d3"], 0.0190814)
    assert np.isclose(svy_mean_without_str_domain.lower_ci["d1"], 0.7902497)
    assert np.isclose(svy_mean_without_str_domain.lower_ci["d2"], 0.7448965)
    assert np.isclose(svy_mean_without_str_domain.lower_ci["d3"], 0.7807285)
    assert np.isclose(svy_mean_without_str_domain.upper_ci["d1"], 0.8720699)
    assert np.isclose(svy_mean_without_str_domain.upper_ci["d2"], 0.8495555)
    assert np.isclose(svy_mean_without_str_domain.upper_ci["d3"], 0.8573434)


def test_mean_estimator_without_str_nor_psu_domain():
    svy_mean_without_str_domain.estimate(y, weight, domain=domain, remove_nan=True)

    assert np.isclose(svy_mean_without_str_domain.point_est["d1"], 0.8311598)
    assert np.isclose(svy_mean_without_str_domain.point_est["d2"], 0.797226)
    assert np.isclose(svy_mean_without_str_domain.point_est["d3"], 0.8190359)
    assert np.isclose(svy_mean_without_str_domain.stderror["d1"], 0.0200196)
    assert np.isclose(svy_mean_without_str_domain.stderror["d2"], 0.0125303)
    assert np.isclose(svy_mean_without_str_domain.stderror["d3"], 0.0085047)
    assert np.isclose(svy_mean_without_str_domain.lower_ci["d1"], 0.7919167)
    assert np.isclose(svy_mean_without_str_domain.lower_ci["d2"], 0.7726636)
    assert np.isclose(svy_mean_without_str_domain.lower_ci["d3"], 0.8023648)
    assert np.isclose(svy_mean_without_str_domain.upper_ci["d1"], 0.8704028)
    assert np.isclose(svy_mean_without_str_domain.upper_ci["d2"], 0.8217885)
    assert np.isclose(svy_mean_without_str_domain.upper_ci["d3"], 0.8357071)


"""Taylor Approximation WITH Stratification for MEAN"""
svy_mean_with_str = TaylorEstimator("mean")


def test_mean_estimator_with_str():
    svy_mean_with_str.estimate(y, weight, stratum=stratum, psu=psu, remove_nan=True)

    assert np.isclose(svy_mean_with_str.point_est, 0.813_622_5)
    assert np.isclose(svy_mean_with_str.variance, 0.019_862_1 ** 2)
    assert np.isclose(svy_mean_with_str.stderror, 0.019_862_1)
    assert np.isclose(svy_mean_with_str.lower_ci, 0.773_447_6)
    assert np.isclose(svy_mean_with_str.upper_ci, 0.853_797_4)
    assert np.isclose(svy_mean_with_str.coef_var, 0.019_862_1 / 0.813_622_5)


def test_mean_estimator_with_str_without_psu():
    svy_mean_with_str.estimate(y, weight, stratum=stratum, remove_nan=True)

    assert np.isclose(svy_mean_with_str.point_est, 0.8136225)
    assert np.isclose(svy_mean_with_str.variance, 0.0066091 ** 2)
    assert np.isclose(svy_mean_with_str.stderror, 0.0066091)
    assert np.isclose(svy_mean_with_str.lower_ci, 0.8006671)
    assert np.isclose(svy_mean_with_str.upper_ci, 0.8265779)
    assert np.isclose(svy_mean_with_str.coef_var, 0.0066091 / 0.8136225)


svy_mean_with_str_domain = TaylorEstimator("mean")


def test_mean_estimator_with_str_domain():
    svy_mean_with_str_domain.estimate(
        y, weight, psu=psu, stratum=stratum, domain=domain, remove_nan=True
    )

    assert np.isclose(svy_mean_with_str_domain.point_est["d1"], 0.8311598)
    assert np.isclose(svy_mean_with_str_domain.point_est["d2"], 0.797226)
    assert np.isclose(svy_mean_with_str_domain.point_est["d3"], 0.8190359)
    assert np.isclose(svy_mean_with_str_domain.stderror["d1"], 0.0200457)
    assert np.isclose(svy_mean_with_str_domain.stderror["d2"], 0.0263015)
    assert np.isclose(svy_mean_with_str_domain.stderror["d3"], 0.0182081)
    assert np.isclose(svy_mean_with_str_domain.lower_ci["d1"], 0.7906136)
    assert np.isclose(svy_mean_with_str_domain.lower_ci["d2"], 0.7440263)
    assert np.isclose(svy_mean_with_str_domain.lower_ci["d3"], 0.7822066)
    assert np.isclose(svy_mean_with_str_domain.upper_ci["d1"], 0.871706)
    assert np.isclose(svy_mean_with_str_domain.upper_ci["d2"], 0.8504258)
    assert np.isclose(svy_mean_with_str_domain.upper_ci["d3"], 0.8558653)


def test_mean_estimator_with_str_nor_psu_domain():
    svy_mean_with_str_domain.estimate(y, weight, stratum=stratum, domain=domain, remove_nan=True)

    assert np.isclose(svy_mean_with_str_domain.point_est["d1"], 0.8311598)
    assert np.isclose(svy_mean_with_str_domain.point_est["d2"], 0.797226)
    assert np.isclose(svy_mean_with_str_domain.point_est["d3"], 0.8190359)
    assert np.isclose(svy_mean_with_str_domain.stderror["d1"], 0.0200198)
    assert np.isclose(svy_mean_with_str_domain.stderror["d2"], 0.0125144)
    assert np.isclose(svy_mean_with_str_domain.stderror["d3"], 0.0084659)
    assert np.isclose(svy_mean_with_str_domain.lower_ci["d1"], 0.7919163)
    assert np.isclose(svy_mean_with_str_domain.lower_ci["d2"], 0.7726948)
    assert np.isclose(svy_mean_with_str_domain.lower_ci["d3"], 0.8024407)
    assert np.isclose(svy_mean_with_str_domain.upper_ci["d1"], 0.8704033)
    assert np.isclose(svy_mean_with_str_domain.upper_ci["d2"], 0.8217573)
    assert np.isclose(svy_mean_with_str_domain.upper_ci["d3"], 0.8356312)


"""Taylor Approximation WITHOUT Stratification for RATIO"""
svy_ratio_without_str = TaylorEstimator("ratio")


def test_ratio_estimator_without_str():
    svy_ratio_without_str.estimate(y, weight, x, psu=psu, remove_nan=True)

    assert np.isclose(svy_ratio_without_str.point_est, 0.706_945_8)
    assert np.isclose(svy_ratio_without_str.variance, 0.007_657_5 ** 2)
    assert np.isclose(svy_ratio_without_str.stderror, 0.007_657_5)
    assert np.isclose(svy_ratio_without_str.lower_ci, 0.691_572_8)
    assert np.isclose(svy_ratio_without_str.upper_ci, 0.722_318_8)
    assert np.isclose(svy_ratio_without_str.coef_var, 0.007_657_5 / 0.706_945_8)


def test_ratio_estimator_without_str_nor_psu():
    svy_ratio_without_str.estimate(y, weight, x, remove_nan=True)

    assert np.isclose(svy_ratio_without_str.point_est, 0.7069458)
    assert np.isclose(svy_ratio_without_str.variance, 0.0025128 ** 2)
    assert np.isclose(svy_ratio_without_str.stderror, 0.0025128)
    assert np.isclose(svy_ratio_without_str.lower_ci, 0.7020202)
    assert np.isclose(svy_ratio_without_str.upper_ci, 0.7118715)
    assert np.isclose(svy_ratio_without_str.coef_var, 0.0025128 / 0.7069458)


svy_ratio_without_str_domain = TaylorEstimator("ratio")


def test_ratio_estimator_without_str_domain():
    svy_ratio_without_str_domain.estimate(y, weight, x, psu=psu, domain=domain, remove_nan=True)

    assert np.isclose(svy_ratio_without_str_domain.point_est["d1"], 0.7134861)
    assert np.isclose(svy_ratio_without_str_domain.point_est["d2"], 0.7006851)
    assert np.isclose(svy_ratio_without_str_domain.point_est["d3"], 0.7089816)
    assert np.isclose(svy_ratio_without_str_domain.stderror["d1"], 0.0075081)
    assert np.isclose(svy_ratio_without_str_domain.stderror["d2"], 0.0100676)
    assert np.isclose(svy_ratio_without_str_domain.stderror["d3"], 0.007149)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci["d1"], 0.698413)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci["d2"], 0.6804736)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci["d3"], 0.6946295)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci["d1"], 0.7285592)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci["d2"], 0.7208966)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci["d3"], 0.7233338)


def test_ratio_estimator_without_str_nor_psu_domain():
    svy_ratio_without_str_domain.estimate(y, weight, x, domain=domain, remove_nan=True)

    assert np.isclose(svy_ratio_without_str_domain.point_est["d1"], 0.7134861)
    assert np.isclose(svy_ratio_without_str_domain.point_est["d2"], 0.7006851)
    assert np.isclose(svy_ratio_without_str_domain.point_est["d3"], 0.7089816)
    assert np.isclose(svy_ratio_without_str_domain.stderror["d1"], 0.0073761)
    assert np.isclose(svy_ratio_without_str_domain.stderror["d2"], 0.0048397)
    assert np.isclose(svy_ratio_without_str_domain.stderror["d3"], 0.0031863)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci["d1"], 0.6990272)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci["d2"], 0.6911982)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci["d3"], 0.7027357)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci["d1"], 0.727945)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci["d2"], 0.710172)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci["d3"], 0.7152276)


"""Taylor Approximation WITH Stratification for RATIO"""
svy_ratio_with_str = TaylorEstimator("ratio")


def test_ratio_estimator_with_str():
    svy_ratio_with_str.estimate(y, weight, x, stratum=stratum, psu=psu, remove_nan=True)

    assert np.isclose(svy_ratio_with_str.point_est, 0.706_945_8)
    assert np.isclose(svy_ratio_with_str.variance, 0.007_497_6 ** 2)
    assert np.isclose(svy_ratio_with_str.stderror, 0.007_497_6)
    assert np.isclose(svy_ratio_with_str.lower_ci, 0.691_780_5)
    assert np.isclose(svy_ratio_with_str.upper_ci, 0.722_111_1)
    assert np.isclose(svy_ratio_with_str.coef_var, 0.007_497_6 / 0.706_945_8)


def test_ratio_estimator_with_str_without_psu():
    svy_ratio_with_str.estimate(y, weight, x, stratum=stratum, remove_nan=True)

    assert np.isclose(svy_ratio_with_str.point_est, 0.7069458)
    assert np.isclose(svy_ratio_with_str.variance, 0.0024948 ** 2)
    assert np.isclose(svy_ratio_with_str.stderror, 0.0024948)
    assert np.isclose(svy_ratio_with_str.lower_ci, 0.7020554)
    assert np.isclose(svy_ratio_with_str.upper_ci, 0.7118362)
    assert np.isclose(svy_ratio_with_str.coef_var, 0.0024948 / 0.7069458)


svy_ratio_with_str_domain = TaylorEstimator("ratio")


def test_ratio_estimator_with_str_domain():
    svy_ratio_with_str_domain.estimate(
        y, weight, x, psu=psu, stratum=stratum, domain=domain, remove_nan=True
    )

    assert np.isclose(svy_ratio_with_str_domain.point_est["d1"], 0.7134861)
    assert np.isclose(svy_ratio_with_str_domain.point_est["d2"], 0.7006851)
    assert np.isclose(svy_ratio_with_str_domain.point_est["d3"], 0.7089816)
    assert np.isclose(svy_ratio_with_str_domain.stderror["d1"], 0.0073857)
    assert np.isclose(svy_ratio_with_str_domain.stderror["d2"], 0.0101586)
    assert np.isclose(svy_ratio_with_str_domain.stderror["d3"], 0.0068218)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci["d1"], 0.6985471)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci["d2"], 0.6801374)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci["d3"], 0.6951852)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci["d1"], 0.7284251)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci["d2"], 0.7212328)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci["d3"], 0.72278)


def test_ratio_estimator_with_str_nor_psu_domain():
    svy_ratio_with_str_domain.estimate(
        y, weight, x, stratum=stratum, domain=domain, remove_nan=True
    )

    assert np.isclose(svy_ratio_with_str_domain.point_est["d1"], 0.7134861)
    assert np.isclose(svy_ratio_with_str_domain.point_est["d2"], 0.7006851)
    assert np.isclose(svy_ratio_with_str_domain.point_est["d3"], 0.7089816)
    assert np.isclose(svy_ratio_with_str_domain.stderror["d1"], 0.0073762)
    assert np.isclose(svy_ratio_with_str_domain.stderror["d2"], 0.0048335)
    assert np.isclose(svy_ratio_with_str_domain.stderror["d3"], 0.0031718)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci["d1"], 0.6990271)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci["d2"], 0.6912102)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci["d3"], 0.7027641)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci["d1"], 0.7279451)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci["d2"], 0.7101599)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci["d3"], 0.7151992)


"""Taylor Approximation WITHOUT Stratification for PROPORTION"""
svy_prop_without_str = TaylorEstimator("proportion")
# svy_prop_without_str.estimate(y, weight, psu=psu, remove_nan=True)
# breakpoint()


def test_prop_estimator_without_str():
    svy_prop_without_str.estimate(y, weight, psu=psu, remove_nan=True)

    assert np.isclose(svy_prop_without_str.point_est[0.0], 0.186_377_5)
    assert np.isclose(svy_prop_without_str.point_est[1.0], 0.813_622_5)
    assert np.isclose(svy_prop_without_str.variance[0.0], 0.020_285_6 ** 2)
    assert np.isclose(svy_prop_without_str.variance[1.0], 0.020_285_6 ** 2)
    assert np.isclose(svy_prop_without_str.stderror[0.0], 0.020_285_6)
    assert np.isclose(svy_prop_without_str.stderror[1.0], 0.020_285_6)
    assert np.isclose(svy_prop_without_str.lower_ci[0.0], 0.149_023)
    assert np.isclose(svy_prop_without_str.lower_ci[1.0], 0.769_441_4)
    assert np.isclose(svy_prop_without_str.upper_ci[0.0], 0.230_558_6)
    assert np.isclose(svy_prop_without_str.upper_ci[1.0], 0.850_977)
    assert np.isclose(svy_prop_without_str.coef_var[0.0], 0.020_285_6 / 0.186_377_5)
    assert np.isclose(svy_prop_without_str.coef_var[1.0], 0.020_285_6 / 0.813_622_5)


def test_prop_estimator_without_str_nor_psu():
    svy_prop_without_str.estimate(y, weight, remove_nan=True)

    assert np.isclose(svy_prop_without_str.point_est[0.0], 0.1863775)
    assert np.isclose(svy_prop_without_str.point_est[1.0], 0.8136225)
    assert np.isclose(svy_prop_without_str.variance[0.0], 0.0066567 ** 2)
    assert np.isclose(svy_prop_without_str.variance[1.0], 0.0066567 ** 2)
    assert np.isclose(svy_prop_without_str.stderror[0.0], 0.0066567)
    assert np.isclose(svy_prop_without_str.stderror[1.0], 0.0066567)
    assert np.isclose(svy_prop_without_str.lower_ci[0.0], 0.1736793)
    assert np.isclose(svy_prop_without_str.lower_ci[1.0], 0.8002204)
    assert np.isclose(svy_prop_without_str.upper_ci[0.0], 0.1997796)
    assert np.isclose(svy_prop_without_str.upper_ci[1.0], 0.8263207)
    assert np.isclose(svy_prop_without_str.coef_var[0.0], 0.0066567 / 0.1863775)
    assert np.isclose(svy_prop_without_str.coef_var[1.0], 0.0066567 / 0.8136225)


svy_prop_without_str_domain = TaylorEstimator("proportion")


def test_prop_estimator_without_str_domain():
    svy_prop_without_str_domain.estimate(y, weight, psu=psu, domain=domain, remove_nan=True)

    assert np.isclose(svy_prop_without_str_domain.point_est["d1"][0.0], 0.1688402)
    assert np.isclose(svy_prop_without_str_domain.point_est["d1"][1.0], 0.8311598)
    assert np.isclose(svy_prop_without_str_domain.point_est["d2"][0.0], 0.202774)
    assert np.isclose(svy_prop_without_str_domain.point_est["d2"][1.0], 0.797226)
    assert np.isclose(svy_prop_without_str_domain.point_est["d3"][0.0], 0.1809641)
    assert np.isclose(svy_prop_without_str_domain.point_est["d3"][1.0], 0.8190359)
    assert np.isclose(svy_prop_without_str_domain.stderror["d1"][0.0], 0.0203778)
    assert np.isclose(svy_prop_without_str_domain.stderror["d1"][1.0], 0.0203778)
    assert np.isclose(svy_prop_without_str_domain.stderror["d2"][0.0], 0.0260659)
    assert np.isclose(svy_prop_without_str_domain.stderror["d2"][1.0], 0.0260659)
    assert np.isclose(svy_prop_without_str_domain.stderror["d3"][0.0], 0.0190814)
    assert np.isclose(svy_prop_without_str_domain.stderror["d3"][1.0], 0.0190814)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d1"][0.0], 0.131771)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d1"][1.0], 0.7862299)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d2"][0.0], 0.155414)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d2"][1.0], 0.7398788)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d3"][0.0], 0.1457555)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d3"][1.0], 0.7775374)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d1"][0.0], 0.2137701)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d1"][1.0], 0.868229)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d2"][0.0], 0.2601212)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d2"][1.0], 0.844586)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d3"][0.0], 0.2224624)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d3"][1.0], 0.8542445)


def test_prop_estimator_without_str_nor_psu_domain():
    svy_prop_without_str_domain.estimate(y, weight, domain=domain, remove_nan=True)

    assert np.isclose(svy_prop_without_str_domain.point_est["d1"][0.0], 0.1688402)
    assert np.isclose(svy_prop_without_str_domain.point_est["d1"][1.0], 0.8311598)
    assert np.isclose(svy_prop_without_str_domain.point_est["d2"][0.0], 0.202774)
    assert np.isclose(svy_prop_without_str_domain.point_est["d2"][1.0], 0.797226)
    assert np.isclose(svy_prop_without_str_domain.point_est["d3"][0.0], 0.1809641)
    assert np.isclose(svy_prop_without_str_domain.point_est["d3"][1.0], 0.8190359)
    assert np.isclose(svy_prop_without_str_domain.stderror["d1"][0.0], 0.0200196)
    assert np.isclose(svy_prop_without_str_domain.stderror["d1"][1.0], 0.0200196)
    assert np.isclose(svy_prop_without_str_domain.stderror["d2"][0.0], 0.0125303)
    assert np.isclose(svy_prop_without_str_domain.stderror["d2"][1.0], 0.0125303)
    assert np.isclose(svy_prop_without_str_domain.stderror["d3"][0.0], 0.0085047)
    assert np.isclose(svy_prop_without_str_domain.stderror["d3"][1.0], 0.0085047)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d1"][0.0], 0.133136)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d1"][1.0], 0.7882197)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d2"][0.0], 0.179316)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d2"][1.0], 0.7715536)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d3"][0.0], 0.1648868)
    assert np.isclose(svy_prop_without_str_domain.lower_ci["d3"][1.0], 0.8017632)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d1"][0.0], 0.2117803)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d1"][1.0], 0.866864)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d2"][0.0], 0.2284464)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d2"][1.0], 0.820684)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d3"][0.0], 0.1982368)
    assert np.isclose(svy_prop_without_str_domain.upper_ci["d3"][1.0], 0.8351132)


"""Taylor Approximation WITH Stratification for PROPORTION"""
svy_prop_with_str = TaylorEstimator("proportion")


def test_prop_estimator_with_str():
    svy_prop_with_str.estimate(y, weight, stratum=stratum, psu=psu, remove_nan=True)

    assert np.isclose(svy_prop_with_str.point_est[0.0], 0.186_377_5)
    assert np.isclose(svy_prop_with_str.point_est[1.0], 0.813_622_5)
    assert np.isclose(svy_prop_with_str.variance[0.0], 0.019_862_1 ** 2)
    assert np.isclose(svy_prop_with_str.variance[1.0], 0.019_862_1 ** 2)
    assert np.isclose(svy_prop_with_str.stderror[0.0], 0.019_862_1)
    assert np.isclose(svy_prop_with_str.stderror[1.0], 0.019_862_1)
    assert np.isclose(svy_prop_with_str.lower_ci[0.0], 0.149_483_7)
    assert np.isclose(svy_prop_with_str.lower_ci[1.0], 0.770_084_5)
    assert np.isclose(svy_prop_with_str.upper_ci[0.0], 0.229_915_5)
    assert np.isclose(svy_prop_with_str.upper_ci[1.0], 0.850_516_3)
    assert np.isclose(svy_prop_with_str.coef_var[0.0], 0.019_862_1 / 0.186_377_5)
    assert np.isclose(svy_prop_with_str.coef_var[1.0], 0.019_862_1 / 0.813_622_5)


def test_prop_estimator_with_str_without_psu():
    svy_prop_with_str.estimate(y, weight, stratum=stratum, remove_nan=True)

    assert np.isclose(svy_prop_with_str.point_est[0.0], 0.1863775)
    assert np.isclose(svy_prop_with_str.point_est[1.0], 0.8136225)
    assert np.isclose(svy_prop_with_str.variance[0.0], 0.0066091 ** 2)
    assert np.isclose(svy_prop_with_str.variance[1.0], 0.0066091 ** 2)
    assert np.isclose(svy_prop_with_str.stderror[0.0], 0.0066091)
    assert np.isclose(svy_prop_with_str.stderror[1.0], 0.0066091)
    assert np.isclose(svy_prop_with_str.lower_ci[0.0], 0.1737677)
    assert np.isclose(svy_prop_with_str.lower_ci[1.0], 0.8003188)
    assert np.isclose(svy_prop_with_str.upper_ci[0.0], 0.1996812)
    assert np.isclose(svy_prop_with_str.upper_ci[1.0], 0.8262323)
    assert np.isclose(svy_prop_with_str.coef_var[0.0], 0.0066091 / 0.1863775)
    assert np.isclose(svy_prop_with_str.coef_var[1.0], 0.0066091 / 0.8136225)


svy_prop_with_str_domain = TaylorEstimator("proportion")


def test_prop_estimator_with_str_domain():
    svy_prop_with_str_domain.estimate(
        y, weight, psu=psu, stratum=stratum, domain=domain, remove_nan=True
    )

    assert np.isclose(svy_prop_with_str_domain.point_est["d1"][0.0], 0.1688402)
    assert np.isclose(svy_prop_with_str_domain.point_est["d1"][1.0], 0.8311598)
    assert np.isclose(svy_prop_with_str_domain.point_est["d2"][0.0], 0.202774)
    assert np.isclose(svy_prop_with_str_domain.point_est["d2"][1.0], 0.797226)
    assert np.isclose(svy_prop_with_str_domain.point_est["d3"][0.0], 0.1809641)
    assert np.isclose(svy_prop_with_str_domain.point_est["d3"][1.0], 0.8190359)
    assert np.isclose(svy_prop_with_str_domain.stderror["d1"][0.0], 0.0200457)
    assert np.isclose(svy_prop_with_str_domain.stderror["d1"][1.0], 0.0200457)
    assert np.isclose(svy_prop_with_str_domain.stderror["d2"][0.0], 0.0263015)
    assert np.isclose(svy_prop_with_str_domain.stderror["d2"][1.0], 0.0263015)
    assert np.isclose(svy_prop_with_str_domain.stderror["d3"][0.0], 0.0182081)
    assert np.isclose(svy_prop_with_str_domain.stderror["d3"][1.0], 0.0182081)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d1"][0.0], 0.1320679)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d1"][1.0], 0.7866654)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d2"][0.0], 0.1547087)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d2"][1.0], 0.7388414)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d3"][0.0], 0.1470016)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d3"][1.0], 0.7792576)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d1"][0.0], 0.2133346)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d1"][1.0], 0.8679321)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d2"][0.0], 0.2611586)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d2"][1.0], 0.8452913)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d3"][0.0], 0.2207424)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d3"][1.0], 0.8529984)


def test_prop_estimator_with_str_nor_psu_domain():
    svy_prop_with_str_domain.estimate(y, weight, stratum=stratum, domain=domain, remove_nan=True)

    assert np.isclose(svy_prop_with_str_domain.point_est["d1"][0.0], 0.1688402)
    assert np.isclose(svy_prop_with_str_domain.point_est["d1"][1.0], 0.8311598)
    assert np.isclose(svy_prop_with_str_domain.point_est["d2"][0.0], 0.202774)
    assert np.isclose(svy_prop_with_str_domain.point_est["d2"][1.0], 0.797226)
    assert np.isclose(svy_prop_with_str_domain.point_est["d3"][0.0], 0.1809641)
    assert np.isclose(svy_prop_with_str_domain.point_est["d3"][1.0], 0.8190359)
    assert np.isclose(svy_prop_with_str_domain.stderror["d1"][0.0], 0.0200198)
    assert np.isclose(svy_prop_with_str_domain.stderror["d1"][1.0], 0.0200198)
    assert np.isclose(svy_prop_with_str_domain.stderror["d2"][0.0], 0.0125144)
    assert np.isclose(svy_prop_with_str_domain.stderror["d2"][1.0], 0.0125144)
    assert np.isclose(svy_prop_with_str_domain.stderror["d3"][0.0], 0.0084659)
    assert np.isclose(svy_prop_with_str_domain.stderror["d3"][1.0], 0.0084659)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d1"][0.0], 0.1331356)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d1"][1.0], 0.7882192)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d2"][0.0], 0.1793444)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d2"][1.0], 0.7715876)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d3"][0.0], 0.1649573)
    assert np.isclose(svy_prop_with_str_domain.lower_ci["d3"][1.0], 0.8018446)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d1"][0.0], 0.2117808)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d1"][1.0], 0.8668644)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d2"][0.0], 0.2284124)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d2"][1.0], 0.8206556)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d3"][0.0], 0.1981554)
    assert np.isclose(svy_prop_with_str_domain.upper_ci["d3"][1.0], 0.8350427)


"""Taylor Approximation WITHOUT Stratification for FACTOR-MEAN"""
svy_factor_mean_without_str = TaylorEstimator("mean")


def test_factor_mean_estimator_without_str():
    svy_factor_mean_without_str.estimate(y, weight, psu=psu, as_factor=True, remove_nan=True)

    assert np.isclose(svy_factor_mean_without_str.point_est[0.0], 0.186_377_5)
    assert np.isclose(svy_factor_mean_without_str.point_est[1.0], 0.813_622_5)
    assert np.isclose(svy_factor_mean_without_str.stderror[0.0], 0.020_285_6)
    assert np.isclose(svy_factor_mean_without_str.stderror[1.0], 0.020_285_6)
    assert np.isclose(svy_factor_mean_without_str.lower_ci[0.0], 0.149_023)
    assert np.isclose(svy_factor_mean_without_str.lower_ci[1.0], 0.769_441_4)
    assert np.isclose(svy_factor_mean_without_str.upper_ci[0.0], 0.230_558_6)
    assert np.isclose(svy_factor_mean_without_str.upper_ci[1.0], 0.850_977)


def test_factor_mean_estimator_without_str_nor_psu():
    svy_factor_mean_without_str.estimate(y, weight, as_factor=True, remove_nan=True)

    assert np.isclose(svy_factor_mean_without_str.point_est[0.0], 0.1863775)
    assert np.isclose(svy_factor_mean_without_str.point_est[1.0], 0.8136225)
    assert np.isclose(svy_factor_mean_without_str.stderror[0.0], 0.0066567)
    assert np.isclose(svy_factor_mean_without_str.stderror[1.0], 0.0066567)
    assert np.isclose(svy_factor_mean_without_str.lower_ci[0.0], 0.1736793)
    assert np.isclose(svy_factor_mean_without_str.lower_ci[1.0], 0.8002204)
    assert np.isclose(svy_factor_mean_without_str.upper_ci[0.0], 0.1997796)
    assert np.isclose(svy_factor_mean_without_str.upper_ci[1.0], 0.8263207)


svy_factor_mean_without_str_domain = TaylorEstimator("mean")


def test_factor_mean_estimator_without_str_domain():
    svy_factor_mean_without_str_domain.estimate(
        y, weight, psu=psu, domain=domain, as_factor=True, remove_nan=True
    )

    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d1"][0.0], 0.1688402)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d1"][1.0], 0.8311598)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d2"][0.0], 0.202774)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d2"][1.0], 0.797226)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d3"][0.0], 0.1809641)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d3"][1.0], 0.8190359)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d1"][0.0], 0.0203778)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d1"][1.0], 0.0203778)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d2"][0.0], 0.0260659)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d2"][1.0], 0.0260659)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d3"][0.0], 0.0190814)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d3"][1.0], 0.0190814)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d1"][0.0], 0.131771)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d1"][1.0], 0.7862299)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d2"][0.0], 0.155414)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d2"][1.0], 0.7398788)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d3"][0.0], 0.1457555)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d3"][1.0], 0.7775374)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d1"][0.0], 0.2137701)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d1"][1.0], 0.868229)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d2"][0.0], 0.2601212)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d2"][1.0], 0.844586)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d3"][0.0], 0.2224624)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d3"][1.0], 0.8542445)


def test_factor_mean_estimator_without_str_nor_psu_domain():
    svy_factor_mean_without_str_domain.estimate(
        y, weight, domain=domain, as_factor=True, remove_nan=True
    )

    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d1"][0.0], 0.1688402)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d1"][1.0], 0.8311598)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d2"][0.0], 0.202774)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d2"][1.0], 0.797226)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d3"][0.0], 0.1809641)
    assert np.isclose(svy_factor_mean_without_str_domain.point_est["d3"][1.0], 0.8190359)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d1"][0.0], 0.0200196)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d1"][1.0], 0.0200196)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d2"][0.0], 0.0125303)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d2"][1.0], 0.0125303)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d3"][0.0], 0.0085047)
    assert np.isclose(svy_factor_mean_without_str_domain.stderror["d3"][1.0], 0.0085047)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d1"][0.0], 0.133136)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d1"][1.0], 0.7882197)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d2"][0.0], 0.179316)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d2"][1.0], 0.7715536)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d3"][0.0], 0.1648868)
    assert np.isclose(svy_factor_mean_without_str_domain.lower_ci["d3"][1.0], 0.8017632)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d1"][0.0], 0.2117803)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d1"][1.0], 0.866864)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d2"][0.0], 0.2284464)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d2"][1.0], 0.820684)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d3"][0.0], 0.1982368)
    assert np.isclose(svy_factor_mean_without_str_domain.upper_ci["d3"][1.0], 0.8351132)


"""Taylor Approximation WITH Stratification for FACTOR-MEAN"""
svy_factor_mean_with_str = TaylorEstimator("mean")


def test_factor_mean_estimator_with_str():
    svy_factor_mean_with_str.estimate(
        y, weight, stratum=stratum, psu=psu, as_factor=True, remove_nan=True
    )

    assert np.isclose(svy_factor_mean_with_str.point_est[0.0], 0.186_377_5)
    assert np.isclose(svy_factor_mean_with_str.point_est[1.0], 0.813_622_5)
    assert np.isclose(svy_factor_mean_with_str.stderror[0.0], 0.019_862_1)
    assert np.isclose(svy_factor_mean_with_str.stderror[1.0], 0.019_862_1)
    assert np.isclose(svy_factor_mean_with_str.lower_ci[0.0], 0.149_483_7)
    assert np.isclose(svy_factor_mean_with_str.lower_ci[1.0], 0.770_084_5)
    assert np.isclose(svy_factor_mean_with_str.upper_ci[0.0], 0.229_915_5)
    assert np.isclose(svy_factor_mean_with_str.upper_ci[1.0], 0.850_516_3)


def test_factor_estimator_with_str_without_psu():
    svy_factor_mean_with_str.estimate(y, weight, stratum=stratum, as_factor=True, remove_nan=True)

    assert np.isclose(svy_factor_mean_with_str.point_est[0.0], 0.1863775)
    assert np.isclose(svy_factor_mean_with_str.point_est[1.0], 0.8136225)
    assert np.isclose(svy_factor_mean_with_str.stderror[0.0], 0.0066091)
    assert np.isclose(svy_factor_mean_with_str.stderror[1.0], 0.0066091)
    assert np.isclose(svy_factor_mean_with_str.lower_ci[0.0], 0.1737677)
    assert np.isclose(svy_factor_mean_with_str.lower_ci[1.0], 0.8003188)
    assert np.isclose(svy_factor_mean_with_str.upper_ci[0.0], 0.1996812)
    assert np.isclose(svy_factor_mean_with_str.upper_ci[1.0], 0.8262323)


svy_factor_mean_with_str_domain = TaylorEstimator("mean")


def test_factor_mean_estimator_with_str_domain():
    svy_factor_mean_with_str_domain.estimate(
        y, weight, psu=psu, stratum=stratum, domain=domain, as_factor=True, remove_nan=True
    )

    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d1"][0.0], 0.1688402)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d1"][1.0], 0.8311598)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d2"][0.0], 0.202774)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d2"][1.0], 0.797226)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d3"][0.0], 0.1809641)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d3"][1.0], 0.8190359)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d1"][0.0], 0.0200457)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d1"][1.0], 0.0200457)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d2"][0.0], 0.0263015)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d2"][1.0], 0.0263015)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d3"][0.0], 0.0182081)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d3"][1.0], 0.0182081)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d1"][0.0], 0.1320679)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d1"][1.0], 0.7866654)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d2"][0.0], 0.1547087)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d2"][1.0], 0.7388414)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d3"][0.0], 0.1470016)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d3"][1.0], 0.7792576)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d1"][0.0], 0.2133346)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d1"][1.0], 0.8679321)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d2"][0.0], 0.2611586)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d2"][1.0], 0.8452913)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d3"][0.0], 0.2207424)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d3"][1.0], 0.8529984)


def test_factor_mean_estimator_with_str_nor_psu_domain():
    svy_factor_mean_with_str_domain.estimate(
        y, weight, stratum=stratum, as_factor=True, domain=domain, remove_nan=True
    )

    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d1"][0.0], 0.1688402)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d1"][1.0], 0.8311598)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d2"][0.0], 0.202774)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d2"][1.0], 0.797226)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d3"][0.0], 0.1809641)
    assert np.isclose(svy_factor_mean_with_str_domain.point_est["d3"][1.0], 0.8190359)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d1"][0.0], 0.0200198)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d1"][1.0], 0.0200198)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d2"][0.0], 0.0125144)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d2"][1.0], 0.0125144)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d3"][0.0], 0.0084659)
    assert np.isclose(svy_factor_mean_with_str_domain.stderror["d3"][1.0], 0.0084659)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d1"][0.0], 0.1331356)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d1"][1.0], 0.7882192)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d2"][0.0], 0.1793444)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d2"][1.0], 0.7715876)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d3"][0.0], 0.1649573)
    assert np.isclose(svy_factor_mean_with_str_domain.lower_ci["d3"][1.0], 0.8018446)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d1"][0.0], 0.2117808)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d1"][1.0], 0.8668644)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d2"][0.0], 0.2284124)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d2"][1.0], 0.8206556)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d3"][0.0], 0.1981554)
    assert np.isclose(svy_factor_mean_with_str_domain.upper_ci["d3"][1.0], 0.8350427)


# """Taylor Approximation WITHOUT Stratification for FACTOR-TOTAL"""
# svy_factor_total_without_str = TaylorEstimator("total")


# def test_factor_total_estimator_without_str():
#     svy_factor_total_without_str.estimate(y, weight, psu=psu, as_factor=True, remove_nan=True)

#     assert np.isclose(svy_factor_total_without_str.point_est[0.0], 0.186_377_5)
#     assert np.isclose(svy_factor_total_without_str.point_est[1.0], 0.813_622_5)
#     assert np.isclose(svy_factor_total_without_str.stderror[0.0], 0.020_285_6)
#     assert np.isclose(svy_factor_total_without_str.stderror[1.0], 0.020_285_6)
#     assert np.isclose(svy_factor_total_without_str.lower_ci[0.0], 0.149_023)
#     assert np.isclose(svy_factor_total_without_str.lower_ci[1.0], 0.769_441_4)
#     assert np.isclose(svy_factor_total_without_str.upper_ci[0.0], 0.230_558_6)
#     assert np.isclose(svy_factor_total_without_str.upper_ci[1.0], 0.850_977)


# def test_factor_total_estimator_without_str_nor_psu():
#     svy_factor_total_without_str.estimate(y, weight, as_factor=True, remove_nan=True)

#     assert np.isclose(svy_factor_total_without_str.point_est[0.0], 0.1863775)
#     assert np.isclose(svy_factor_total_without_str.point_est[1.0], 0.8136225)
#     assert np.isclose(svy_factor_total_without_str.stderror[0.0], 0.0066567)
#     assert np.isclose(svy_factor_total_without_str.stderror[1.0], 0.0066567)
#     assert np.isclose(svy_factor_total_without_str.lower_ci[0.0], 0.1736793)
#     assert np.isclose(svy_factor_total_without_str.lower_ci[1.0], 0.8002204)
#     assert np.isclose(svy_factor_total_without_str.upper_ci[0.0], 0.1997796)
#     assert np.isclose(svy_factor_total_without_str.upper_ci[1.0], 0.8263207)


# svy_factor_total_without_str_domain = TaylorEstimator("total")


# def test_factor_total_estimator_without_str_domain():
#     svy_factor_total_without_str_domain.estimate(
#         y, weight, psu=psu, domain=domain, as_factor=True, remove_nan=True
#     )

#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d1"][0.0], 0.1688402)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d1"][1.0], 0.8311598)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d2"][0.0], 0.202774)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d2"][1.0], 0.797226)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d3"][0.0], 0.1809641)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d3"][1.0], 0.8190359)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d1"][0.0], 0.0203778)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d1"][1.0], 0.0203778)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d2"][0.0], 0.0260659)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d2"][1.0], 0.0260659)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d3"][0.0], 0.0190814)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d3"][1.0], 0.0190814)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d1"][0.0], 0.131771)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d1"][1.0], 0.7862299)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d2"][0.0], 0.155414)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d2"][1.0], 0.7398788)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d3"][0.0], 0.1457555)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d3"][1.0], 0.7775374)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d1"][0.0], 0.2137701)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d1"][1.0], 0.868229)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d2"][0.0], 0.2601212)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d2"][1.0], 0.844586)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d3"][0.0], 0.2224624)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d3"][1.0], 0.8542445)


# def test_factor_total_estimator_without_str_nor_psu_domain():
#     svy_factor_total_without_str_domain.estimate(
#         y, weight, domain=domain, as_factor=True, remove_nan=True
#     )

#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d1"][0.0], 0.1688402)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d1"][1.0], 0.8311598)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d2"][0.0], 0.202774)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d2"][1.0], 0.797226)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d3"][0.0], 0.1809641)
#     assert np.isclose(svy_factor_total_without_str_domain.point_est["d3"][1.0], 0.8190359)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d1"][0.0], 0.0200196)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d1"][1.0], 0.0200196)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d2"][0.0], 0.0125303)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d2"][1.0], 0.0125303)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d3"][0.0], 0.0085047)
#     assert np.isclose(svy_factor_total_without_str_domain.stderror["d3"][1.0], 0.0085047)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d1"][0.0], 0.133136)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d1"][1.0], 0.7882197)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d2"][0.0], 0.179316)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d2"][1.0], 0.7715536)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d3"][0.0], 0.1648868)
#     assert np.isclose(svy_factor_total_without_str_domain.lower_ci["d3"][1.0], 0.8017632)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d1"][0.0], 0.2117803)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d1"][1.0], 0.866864)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d2"][0.0], 0.2284464)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d2"][1.0], 0.820684)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d3"][0.0], 0.1982368)
#     assert np.isclose(svy_factor_total_without_str_domain.upper_ci["d3"][1.0], 0.8351132)


# """Taylor Approximation WITH Stratification for FACTOR-TOTAL"""
# svy_factor_total_with_str = TaylorEstimator("total")


# def test_factor_total_estimator_with_str():
#     svy_factor_total_with_str.estimate(
#         y, weight, stratum=stratum, psu=psu, as_factor=True, remove_nan=True
#     )

#     assert np.isclose(svy_factor_total_with_str.point_est[0.0], 0.186_377_5)
#     assert np.isclose(svy_factor_total_with_str.point_est[1.0], 0.813_622_5)
#     assert np.isclose(svy_factor_total_with_str.stderror[0.0], 0.019_862_1)
#     assert np.isclose(svy_factor_total_with_str.stderror[1.0], 0.019_862_1)
#     assert np.isclose(svy_factor_total_with_str.lower_ci[0.0], 0.149_483_7)
#     assert np.isclose(svy_factor_total_with_str.lower_ci[1.0], 0.770_084_5)
#     assert np.isclose(svy_factor_total_with_str.upper_ci[0.0], 0.229_915_5)
#     assert np.isclose(svy_factor_total_with_str.upper_ci[1.0], 0.850_516_3)


# def test_factor_estimator_with_str_without_psu():
#     svy_factor_total_with_str.estimate(y, weight, stratum=stratum, as_factor=True, remove_nan=True)

#     assert np.isclose(svy_factor_total_with_str.point_est[0.0], 0.1863775)
#     assert np.isclose(svy_factor_total_with_str.point_est[1.0], 0.8136225)
#     assert np.isclose(svy_factor_total_with_str.stderror[0.0], 0.0066091)
#     assert np.isclose(svy_factor_total_with_str.stderror[1.0], 0.0066091)
#     assert np.isclose(svy_factor_total_with_str.lower_ci[0.0], 0.1737677)
#     assert np.isclose(svy_factor_total_with_str.lower_ci[1.0], 0.8003188)
#     assert np.isclose(svy_factor_total_with_str.upper_ci[0.0], 0.1996812)
#     assert np.isclose(svy_factor_total_with_str.upper_ci[1.0], 0.8262323)


# svy_factor_total_with_str_domain = TaylorEstimator("total")


# def test_factor_total_estimator_with_str_domain():
#     svy_factor_total_with_str_domain.estimate(
#         y, weight, psu=psu, stratum=stratum, domain=domain, as_factor=True, remove_nan=True
#     )

#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d1"][0.0], 0.1688402)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d1"][1.0], 0.8311598)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d2"][0.0], 0.202774)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d2"][1.0], 0.797226)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d3"][0.0], 0.1809641)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d3"][1.0], 0.8190359)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d1"][0.0], 0.0200457)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d1"][1.0], 0.0200457)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d2"][0.0], 0.0263015)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d2"][1.0], 0.0263015)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d3"][0.0], 0.0182081)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d3"][1.0], 0.0182081)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d1"][0.0], 0.1320679)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d1"][1.0], 0.7866654)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d2"][0.0], 0.1547087)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d2"][1.0], 0.7388414)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d3"][0.0], 0.1470016)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d3"][1.0], 0.7792576)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d1"][0.0], 0.2133346)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d1"][1.0], 0.8679321)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d2"][0.0], 0.2611586)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d2"][1.0], 0.8452913)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d3"][0.0], 0.2207424)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d3"][1.0], 0.8529984)


# def test_factor_total_estimator_with_str_nor_psu_domain():
#     svy_factor_total_with_str_domain.estimate(
#         y, weight, stratum=stratum, as_factor=True, domain=domain, remove_nan=True
#     )

#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d1"][0.0], 0.1688402)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d1"][1.0], 0.8311598)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d2"][0.0], 0.202774)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d2"][1.0], 0.797226)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d3"][0.0], 0.1809641)
#     assert np.isclose(svy_factor_total_with_str_domain.point_est["d3"][1.0], 0.8190359)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d1"][0.0], 0.0200198)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d1"][1.0], 0.0200198)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d2"][0.0], 0.0125144)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d2"][1.0], 0.0125144)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d3"][0.0], 0.0084659)
#     assert np.isclose(svy_factor_total_with_str_domain.stderror["d3"][1.0], 0.0084659)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d1"][0.0], 0.1331356)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d1"][1.0], 0.7882192)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d2"][0.0], 0.1793444)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d2"][1.0], 0.7715876)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d3"][0.0], 0.1649573)
#     assert np.isclose(svy_factor_total_with_str_domain.lower_ci["d3"][1.0], 0.8018446)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d1"][0.0], 0.2117808)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d1"][1.0], 0.8668644)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d2"][0.0], 0.2284124)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d2"][1.0], 0.8206556)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d3"][0.0], 0.1981554)
#     assert np.isclose(svy_factor_total_with_str_domain.upper_ci["d3"][1.0], 0.8350427)


"""Testing conversion to DataFrame"""
svy_est = TaylorEstimator("mean")


def test_factor_mean_estimator_with_str_dataframe():
    svy_est.estimate(y, weight, stratum=stratum, psu=psu, domain=domain, remove_nan=True)
    svy_est_df = svy_est.to_dataframe()

    assert svy_est_df.columns.tolist() == [
        "_parameter",
        "_domain",
        "_level",
        "_estimate",
        "_stderror",
        "_lci",
        "_uci",
        "_cv",
    ]

    assert svy_est_df.iloc[0, 0] == "mean"
    assert svy_est_df.iloc[1, 0] == "mean"
    assert svy_est_df.iloc[2, 0] == "mean"
    assert svy_est_df.iloc[0, 1] == "d1"
    assert svy_est_df.iloc[1, 1] == "d2"
    assert svy_est_df.iloc[2, 1] == "d3"
    assert np.isclose(svy_est_df.iloc[0, 2], 0.831160, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 2], 0.797226, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 2], 0.819036, 1e-4)
    assert np.isclose(svy_est_df.iloc[0, 3], 0.020046, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 3], 0.026301, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 3], 0.018208, 1e-4)
    assert np.isclose(svy_est_df.iloc[0, 4], 0.790614, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 4], 0.744026, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 4], 0.782207, 1e-4)
    assert np.isclose(svy_est_df.iloc[0, 5], 0.871706, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 5], 0.850426, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 5], 0.855865, 1e-4)
    assert np.isclose(svy_est_df.iloc[0, 6], 0.024118, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 6], 0.032991, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 6], 0.022231, 1e-4)


"""Testing conversion to DataFrame with as_factor=True"""
svy_est = TaylorEstimator("mean")


def test_factor_mean_estimator_with_str_dataframe_as_factor():
    svy_est.estimate(
        y, weight, stratum=stratum, psu=psu, domain=domain, as_factor=True, remove_nan=True
    )
    svy_est_df = svy_est.to_dataframe()

    assert svy_est_df.columns.tolist() == [
        "_parameter",
        "_domain",
        "_level",
        "_estimate",
        "_stderror",
        "_lci",
        "_uci",
        "_cv",
    ]
    assert svy_est_df.iloc[0, 0] == "mean"
    assert svy_est_df.iloc[1, 0] == "mean"
    assert svy_est_df.iloc[2, 0] == "mean"
    assert svy_est_df.iloc[3, 0] == "mean"
    assert svy_est_df.iloc[4, 0] == "mean"
    assert svy_est_df.iloc[5, 0] == "mean"
    assert svy_est_df.iloc[0, 1] == "d1"
    assert svy_est_df.iloc[1, 1] == "d1"
    assert svy_est_df.iloc[2, 1] == "d2"
    assert svy_est_df.iloc[3, 1] == "d2"
    assert svy_est_df.iloc[4, 1] == "d3"
    assert svy_est_df.iloc[5, 1] == "d3"
    # assert svy_est_df.iloc[0, 2] == ""
    # assert svy_est_df.iloc[1, 2] == ""
    # assert svy_est_df.iloc[2, 2] == ""
    # assert svy_est_df.iloc[3, 2] == ""
    # assert svy_est_df.iloc[4, 2] == ""
    # assert svy_est_df.iloc[5, 2] == ""
    assert np.isclose(svy_est_df.iloc[0, 3], 0.168840, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 3], 0.831160, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 3], 0.202774, 1e-4)
    assert np.isclose(svy_est_df.iloc[3, 3], 0.797226, 1e-4)
    assert np.isclose(svy_est_df.iloc[4, 3], 0.180964, 1e-4)
    assert np.isclose(svy_est_df.iloc[5, 3], 0.819036, 1e-4)
    assert np.isclose(svy_est_df.iloc[0, 4], 0.020046, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 4], 0.020046, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 4], 0.026301, 1e-4)
    assert np.isclose(svy_est_df.iloc[3, 4], 0.026301, 1e-4)
    assert np.isclose(svy_est_df.iloc[4, 4], 0.018208, 1e-4)
    assert np.isclose(svy_est_df.iloc[5, 4], 0.018208, 1e-4)
    assert np.isclose(svy_est_df.iloc[0, 5], 0.132068, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 5], 0.786665, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 5], 0.154709, 1e-4)
    assert np.isclose(svy_est_df.iloc[3, 5], 0.738841, 1e-4)
    assert np.isclose(svy_est_df.iloc[4, 5], 0.147002, 1e-4)
    assert np.isclose(svy_est_df.iloc[5, 5], 0.779258, 1e-4)
    assert np.isclose(svy_est_df.iloc[0, 6], 0.213335, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 6], 0.867932, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 6], 0.261159, 1e-4)
    assert np.isclose(svy_est_df.iloc[3, 6], 0.845291, 1e-4)
    assert np.isclose(svy_est_df.iloc[4, 6], 0.220742, 1e-4)
    assert np.isclose(svy_est_df.iloc[5, 6], 0.852998, 1e-4)
    assert np.isclose(svy_est_df.iloc[0, 7], 0.118726, 1e-4)
    assert np.isclose(svy_est_df.iloc[1, 7], 0.024118, 1e-4)
    assert np.isclose(svy_est_df.iloc[2, 7], 0.129708, 1e-4)
    assert np.isclose(svy_est_df.iloc[3, 7], 0.032991, 1e-4)
    assert np.isclose(svy_est_df.iloc[4, 7], 0.100617, 1e-4)
    assert np.isclose(svy_est_df.iloc[5, 7], 0.022231, 1e-4)