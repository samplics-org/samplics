import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator

one_stage_fpc = pd.read_csv("./tests/estimation/one_stage_fpc.csv")

fpc_array_psu = 1 - one_stage_fpc["nh"] / one_stage_fpc["Nh"]
fpc_dict_psu = dict(zip(one_stage_fpc["stratid"], fpc_array_psu))
fpc_dict_no_psu = {1: 1 - 5 / 15, 2: 1 - 6 / 9, 3: 1 - 10 / 20}

stratum = one_stage_fpc["stratid"]
psu = one_stage_fpc["psuid"]
weight = one_stage_fpc["weight"]
domain = one_stage_fpc["domain"]
x = one_stage_fpc["x"]
y = one_stage_fpc["y"]
fpc_psu = 1 - 5 / 44
fpc_no_psu = 1 - 21 / 44

"""Taylor Approximation WITHOUT Stratification for TOTAL"""
svy_total_without_str = TaylorEstimator("total")


def test_total_estimator_without_str():
    svy_total_without_str.estimate(y, weight, psu=psu, fpc=fpc_psu)

    assert np.isclose(svy_total_without_str.point_est["__none__"], 853.12)
    assert np.isclose(svy_total_without_str.stderror["__none__"], 153.2627)
    assert np.isclose(svy_total_without_str.lower_ci["__none__"], 427.5946)
    assert np.isclose(svy_total_without_str.upper_ci["__none__"], 1278.645)


def test_total_estimator_without_str_nor_psu():
    svy_total_without_str.estimate(y, weight, fpc=fpc_no_psu)

    assert np.isclose(svy_total_without_str.point_est["__none__"], 853.12)
    assert np.isclose(svy_total_without_str.stderror["__none__"], 67.14785)
    assert np.isclose(svy_total_without_str.lower_ci["__none__"], 713.052)
    assert np.isclose(svy_total_without_str.upper_ci["__none__"], 993.188)


svy_total_without_str_domain = TaylorEstimator("total")


def test_total_estimator_without_str_domain():
    svy_total_without_str_domain.estimate(y, weight, psu=psu, fpc=fpc_psu, domain=domain)

    assert np.isclose(svy_total_without_str_domain.point_est[1], 528.52)
    assert np.isclose(svy_total_without_str_domain.point_est[2], 324.6)
    assert np.isclose(svy_total_without_str_domain.stderror[1], 195.5778)
    assert np.isclose(svy_total_without_str_domain.stderror[2], 116.0081)
    assert np.isclose(svy_total_without_str_domain.lower_ci[1], -14.49112)
    assert np.isclose(svy_total_without_str_domain.lower_ci[2], 2.510021)
    assert np.isclose(svy_total_without_str_domain.upper_ci[1], 1071.531)
    assert np.isclose(svy_total_without_str_domain.upper_ci[2], 646.69)


def test_total_estimator_without_str_nor_psu_domain():
    svy_total_without_str_domain.estimate(y, weight, domain=domain, fpc=fpc_no_psu)

    assert np.isclose(svy_total_without_str_domain.point_est[1], 528.52)
    assert np.isclose(svy_total_without_str_domain.point_est[2], 324.6)
    assert np.isclose(svy_total_without_str_domain.stderror[1], 92.9434)
    assert np.isclose(svy_total_without_str_domain.stderror[2], 69.55674)
    assert np.isclose(svy_total_without_str_domain.lower_ci[1], 334.6435)
    assert np.isclose(svy_total_without_str_domain.lower_ci[2], 179.5072)
    assert np.isclose(svy_total_without_str_domain.upper_ci[1], 722.3965)
    assert np.isclose(svy_total_without_str_domain.upper_ci[2], 469.6928)


"""Taylor Approximation WITH Stratification for TOTAL"""
svy_total_with_str = TaylorEstimator("total")


def test_total_estimator_with_str():
    svy_total_with_str.estimate(y, weight, stratum=stratum, psu=psu, fpc=fpc_dict_psu)

    assert np.isclose(svy_total_with_str.point_est["__none__"], 853.12)
    assert np.isclose(svy_total_with_str.stderror["__none__"], 97.41386)
    assert np.isclose(svy_total_with_str.lower_ci["__none__"], 632.75452)
    assert np.isclose(svy_total_with_str.upper_ci["__none__"], 1073.48547)


def test_total_estimator_with_str_without_psu():
    svy_total_with_str.estimate(y, weight, stratum=stratum, fpc=fpc_dict_no_psu)

    assert np.isclose(svy_total_with_str.point_est["__none__"], 853.12)
    assert np.isclose(svy_total_with_str.stderror["__none__"], 50.8888)
    assert np.isclose(svy_total_with_str.lower_ci["__none__"], 746.2066)
    assert np.isclose(svy_total_with_str.upper_ci["__none__"], 960.0334)


svy_total_with_str_domain = TaylorEstimator("total")


def test_total_estimator_with_str_domain():
    svy_total_with_str_domain.estimate(
        y, weight, stratum=stratum, psu=psu, domain=domain, fpc=fpc_dict_psu
    )

    assert np.isclose(svy_total_with_str_domain.point_est[1], 528.52)
    assert np.isclose(svy_total_with_str_domain.point_est[2], 324.6)
    assert np.isclose(svy_total_with_str_domain.stderror[1], 165.5088)
    assert np.isclose(svy_total_with_str_domain.stderror[2], 97.16214)
    assert np.isclose(svy_total_with_str_domain.lower_ci[1], 154.113)
    assert np.isclose(svy_total_with_str_domain.lower_ci[2], 104.804)
    assert np.isclose(svy_total_with_str_domain.upper_ci[1], 902.927)
    assert np.isclose(svy_total_with_str_domain.upper_ci[2], 544.396)


def test_total_estimator_with_str_without_psu_domain():
    svy_total_with_str_domain.estimate(
        y, weight, stratum=stratum, domain=domain, fpc=fpc_dict_no_psu
    )

    assert np.isclose(svy_total_with_str_domain.point_est[1], 528.52)
    assert np.isclose(svy_total_with_str_domain.point_est[2], 324.6)
    assert np.isclose(svy_total_with_str_domain.stderror[1], 80.33344)
    assert np.isclose(svy_total_with_str_domain.stderror[2], 72.63578)
    assert np.isclose(svy_total_with_str_domain.lower_ci[1], 359.7457)
    assert np.isclose(svy_total_with_str_domain.lower_ci[2], 171.9979)
    assert np.isclose(svy_total_with_str_domain.upper_ci[1], 697.2943)
    assert np.isclose(svy_total_with_str_domain.upper_ci[2], 477.2021)


"""Taylor Approximation WITHOUT Stratification for MEAN"""
svy_mean_without_str = TaylorEstimator("mean")


def test_mean_estimator_without_str():
    svy_mean_without_str.estimate(y, weight, psu=psu, fpc=fpc_psu)

    assert np.isclose(svy_mean_without_str.point_est["__none__"], 10.27855)
    assert np.isclose(svy_mean_without_str.stderror["__none__"], 1.126881)
    assert np.isclose(svy_mean_without_str.lower_ci["__none__"], 7.14983)
    assert np.isclose(svy_mean_without_str.upper_ci["__none__"], 13.40728)


def test_mean_estimator_without_str_nor_psu():
    svy_mean_without_str.estimate(y, weight, fpc=fpc_no_psu)

    assert np.isclose(svy_mean_without_str.point_est["__none__"], 10.27855)
    assert np.isclose(svy_mean_without_str.stderror["__none__"], 0.6177327)
    assert np.isclose(svy_mean_without_str.lower_ci["__none__"], 8.989986)
    assert np.isclose(svy_mean_without_str.upper_ci["__none__"], 11.56712)


svy_mean_without_str_domain = TaylorEstimator("mean")


def test_mean_estimator_without_str_domain():
    svy_mean_without_str_domain.estimate(y, weight, psu=psu, fpc=fpc_psu, domain=domain)

    assert np.isclose(svy_mean_without_str_domain.point_est[1], 9.972075)
    assert np.isclose(svy_mean_without_str_domain.point_est[2], 10.82)
    assert np.isclose(svy_mean_without_str_domain.stderror[1], 1.481254)
    assert np.isclose(svy_mean_without_str_domain.stderror[2], 1.30148)
    assert np.isclose(svy_mean_without_str_domain.lower_ci[1], 5.859456)
    assert np.isclose(svy_mean_without_str_domain.lower_ci[2], 7.206513)
    assert np.isclose(svy_mean_without_str_domain.upper_ci[1], 14.08469)
    assert np.isclose(svy_mean_without_str_domain.upper_ci[2], 14.43349)


def test_mean_estimator_without_str_nor_psu_domain():
    svy_mean_without_str_domain.estimate(y, weight, domain=domain, fpc=fpc_no_psu)

    assert np.isclose(svy_mean_without_str_domain.point_est[1], 9.972075)
    assert np.isclose(svy_mean_without_str_domain.point_est[2], 10.82)
    assert np.isclose(svy_mean_without_str_domain.stderror[1], 0.9018966)
    assert np.isclose(svy_mean_without_str_domain.stderror[2], 0.6436988)
    assert np.isclose(svy_mean_without_str_domain.lower_ci[1], 8.090752)
    assert np.isclose(svy_mean_without_str_domain.lower_ci[2], 9.477268)
    assert np.isclose(svy_mean_without_str_domain.upper_ci[1], 11.8534)
    assert np.isclose(svy_mean_without_str_domain.upper_ci[2], 12.16273)


"""Taylor Approximation WITH Stratification for MEAN"""
svy_mean_with_str = TaylorEstimator("mean")


def test_mean_estimator_with_str():
    svy_mean_with_str.estimate(y, weight, stratum=stratum, psu=psu, fpc=fpc_dict_psu)

    assert np.isclose(svy_mean_with_str.point_est["__none__"], 10.27855)
    assert np.isclose(svy_mean_with_str.stderror["__none__"], 0.7750964)
    assert np.isclose(svy_mean_with_str.lower_ci["__none__"], 8.525164)
    assert np.isclose(svy_mean_with_str.upper_ci["__none__"], 12.03194)


def test_mean_estimator_with_str_without_psu():
    svy_mean_with_str.estimate(y, weight, stratum=stratum, fpc=fpc_dict_no_psu)

    assert np.isclose(svy_mean_with_str.point_est["__none__"], 10.27855)
    assert np.isclose(svy_mean_with_str.stderror["__none__"], 0.6131181)
    assert np.isclose(svy_mean_with_str.lower_ci["__none__"], 8.990441)
    assert np.isclose(svy_mean_with_str.upper_ci["__none__"], 11.56667)


svy_mean_with_str_domain = TaylorEstimator("mean")


def test_mean_estimator_with_str_domain():
    svy_mean_with_str_domain.estimate(
        y, weight, stratum=stratum, psu=psu, domain=domain, fpc=fpc_dict_psu
    )

    assert np.isclose(svy_mean_with_str_domain.point_est[1], 9.972075)
    assert np.isclose(svy_mean_with_str_domain.point_est[2], 10.82)
    assert np.isclose(svy_mean_with_str_domain.stderror[1], 1.05736)
    assert np.isclose(svy_mean_with_str_domain.stderror[2], 0.9217838)
    assert np.isclose(svy_mean_with_str_domain.lower_ci[1], 7.580162)
    assert np.isclose(svy_mean_with_str_domain.lower_ci[2], 8.73478)
    assert np.isclose(svy_mean_with_str_domain.upper_ci[1], 12.36399)
    assert np.isclose(svy_mean_with_str_domain.upper_ci[2], 12.90522)


def test_mean_estimator_with_str_nor_psu_domain():
    svy_mean_with_str_domain.estimate(
        y, weight, stratum=stratum, domain=domain, fpc=fpc_dict_no_psu
    )

    assert np.isclose(svy_mean_with_str_domain.point_est[1], 9.972075)
    assert np.isclose(svy_mean_with_str_domain.point_est[2], 10.82)
    assert np.isclose(svy_mean_with_str_domain.stderror[1], 0.8437669)
    assert np.isclose(svy_mean_with_str_domain.stderror[2], 0.6267699)
    assert np.isclose(svy_mean_with_str_domain.lower_ci[1], 8.199387)
    assert np.isclose(svy_mean_with_str_domain.lower_ci[2], 9.503206)
    assert np.isclose(svy_mean_with_str_domain.upper_ci[1], 11.74476)
    assert np.isclose(svy_mean_with_str_domain.upper_ci[2], 12.13679)


"""Taylor Approximation WITHOUT Stratification for RATIO"""
svy_ratio_without_str = TaylorEstimator("ratio")


def test_ratio_estimator_without_str():
    svy_ratio_without_str.estimate(y, weight, x, psu=psu, fpc=fpc_psu)

    assert np.isclose(svy_ratio_without_str.point_est["__none__"], 1.892038)
    assert np.isclose(svy_ratio_without_str.stderror["__none__"], 0.0106065)
    assert np.isclose(svy_ratio_without_str.lower_ci["__none__"], 1.86259)
    assert np.isclose(svy_ratio_without_str.upper_ci["__none__"], 1.921486)


def test_ratio_estimator_without_str_nor_psu():
    svy_ratio_without_str.estimate(y, weight, x, fpc=fpc_no_psu)

    assert np.isclose(svy_ratio_without_str.point_est["__none__"], 1.892038)
    assert np.isclose(svy_ratio_without_str.stderror["__none__"], 0.0058142)
    assert np.isclose(svy_ratio_without_str.lower_ci["__none__"], 1.87991)
    assert np.isclose(svy_ratio_without_str.upper_ci["__none__"], 1.904166)


svy_ratio_without_str_domain = TaylorEstimator("ratio")


def test_ratio_estimator_without_str_domain():
    svy_ratio_without_str_domain.estimate(y, weight, x, psu=psu, fpc=fpc_psu, domain=domain)

    assert np.isclose(svy_ratio_without_str_domain.point_est[1], 1.8950016)
    assert np.isclose(svy_ratio_without_str_domain.point_est[2], 1.887209)
    assert np.isclose(svy_ratio_without_str_domain.stderror[1], 0.0148587)
    assert np.isclose(svy_ratio_without_str_domain.stderror[2], 0.0109982)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci[1], 1.853762)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci[2], 1.856674)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci[1], 1.936271)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci[2], 1.917745)


def test_ratio_estimator_without_str_nor_psu_domain():
    svy_ratio_without_str_domain.estimate(y, weight, x, domain=domain, fpc=fpc_no_psu)

    assert np.isclose(svy_ratio_without_str_domain.point_est[1], 1.895016)
    assert np.isclose(svy_ratio_without_str_domain.point_est[2], 1.887209)
    assert np.isclose(svy_ratio_without_str_domain.stderror[1], 0.0090471)
    assert np.isclose(svy_ratio_without_str_domain.stderror[2], 0.0054396)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci[1], 1.876144)
    assert np.isclose(svy_ratio_without_str_domain.lower_ci[2], 1.875863)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci[1], 1.913888)
    assert np.isclose(svy_ratio_without_str_domain.upper_ci[2], 1.898556)


# """Taylor Approximation WITH Stratification for RATIO"""
svy_ratio_with_str = TaylorEstimator("ratio")


def test_ratio_estimator_with_str():
    svy_ratio_with_str.estimate(y, weight, x, stratum=stratum, psu=psu, fpc=fpc_dict_psu)

    assert np.isclose(svy_ratio_with_str.point_est["__none__"], 1.892038)
    assert np.isclose(svy_ratio_with_str.stderror["__none__"], 0.0072954)
    assert np.isclose(svy_ratio_with_str.lower_ci["__none__"], 1.875535)
    assert np.isclose(svy_ratio_with_str.upper_ci["__none__"], 1.908542)


def test_ratio_estimator_with_str_without_psu():
    svy_ratio_with_str.estimate(y, weight, x, stratum=stratum, fpc=fpc_dict_no_psu)

    assert np.isclose(svy_ratio_with_str.point_est["__none__"], 1.892038)
    assert np.isclose(svy_ratio_with_str.stderror["__none__"], 0.0057708)
    assert np.isclose(svy_ratio_with_str.lower_ci["__none__"], 1.879914)
    assert np.isclose(svy_ratio_with_str.upper_ci["__none__"], 1.904162)


svy_ratio_with_str_domain = TaylorEstimator("ratio")


def test_ratio_estimator_with_str_domain():
    svy_ratio_with_str_domain.estimate(
        y, weight, x, stratum=stratum, psu=psu, domain=domain, fpc=fpc_dict_psu
    )

    assert np.isclose(svy_ratio_with_str_domain.point_est[1], 1.895016)
    assert np.isclose(svy_ratio_with_str_domain.point_est[2], 1.887209)
    assert np.isclose(svy_ratio_with_str_domain.stderror[1], 0.0106066)
    assert np.isclose(svy_ratio_with_str_domain.stderror[2], 0.0077895)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci[1], 1.871022)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci[2], 1.869588)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci[1], 1.91901)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci[2], 1.90483)


def test_ratio_estimator_with_str_nor_psu_domain():
    svy_ratio_with_str_domain.estimate(
        y, weight, x, stratum=stratum, domain=domain, fpc=fpc_dict_no_psu
    )

    assert np.isclose(svy_ratio_with_str_domain.point_est[1], 1.895016)
    assert np.isclose(svy_ratio_with_str_domain.point_est[2], 1.887209)
    assert np.isclose(svy_ratio_with_str_domain.stderror[1], 0.008464)
    assert np.isclose(svy_ratio_with_str_domain.stderror[2], 0.0052965)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci[1], 1.877234)
    assert np.isclose(svy_ratio_with_str_domain.lower_ci[2], 1.876082)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci[1], 1.912798)
    assert np.isclose(svy_ratio_with_str_domain.upper_ci[2], 1.898337)


# """Taylor Approximation WITHOUT Stratification for PROPORTION"""
# svy_prop_without_str = TaylorEstimator("proportion")


# def test_prop_estimator_without_str():
#     svy_prop_without_str.estimate(y, weight, psu=psu, remove_nan=True)

#     assert np.isclose(svy_prop_without_str.point_est["__none__"][0.0], 0.186_377_5)
#     assert np.isclose(svy_prop_without_str.point_est["__none__"][1.0], 0.813_622_5)
#     assert np.isclose(svy_prop_without_str.variance["__none__"][0.0], 0.020_285_6 ** 2)
#     assert np.isclose(svy_prop_without_str.variance["__none__"][1.0], 0.020_285_6 ** 2)
#     assert np.isclose(svy_prop_without_str.stderror["__none__"][0.0], 0.020_285_6)
#     assert np.isclose(svy_prop_without_str.stderror["__none__"][1.0], 0.020_285_6)
#     assert np.isclose(svy_prop_without_str.lower_ci["__none__"][0.0], 0.149_023)
#     assert np.isclose(svy_prop_without_str.lower_ci["__none__"][1.0], 0.769_441_4)
#     assert np.isclose(svy_prop_without_str.upper_ci["__none__"][0.0], 0.230_558_6)
#     assert np.isclose(svy_prop_without_str.upper_ci["__none__"][1.0], 0.850_977)
#     assert np.isclose(svy_prop_without_str.coef_var["__none__"][0.0], 0.020_285_6 / 0.186_377_5)
#     assert np.isclose(svy_prop_without_str.coef_var["__none__"][1.0], 0.020_285_6 / 0.813_622_5)


# def test_prop_estimator_without_str_nor_psu():
#     svy_prop_without_str.estimate(y, weight, remove_nan=True)

#     assert np.isclose(svy_prop_without_str.point_est["__none__"][0.0], 0.1863775)
#     assert np.isclose(svy_prop_without_str.point_est["__none__"][1.0], 0.8136225)
#     assert np.isclose(svy_prop_without_str.variance["__none__"][0.0], 0.0066567 ** 2)
#     assert np.isclose(svy_prop_without_str.variance["__none__"][1.0], 0.0066567 ** 2)
#     assert np.isclose(svy_prop_without_str.stderror["__none__"][0.0], 0.0066567)
#     assert np.isclose(svy_prop_without_str.stderror["__none__"][1.0], 0.0066567)
#     assert np.isclose(svy_prop_without_str.lower_ci["__none__"][0.0], 0.1736793)
#     assert np.isclose(svy_prop_without_str.lower_ci["__none__"][1.0], 0.8002204)
#     assert np.isclose(svy_prop_without_str.upper_ci["__none__"][0.0], 0.1997796)
#     assert np.isclose(svy_prop_without_str.upper_ci["__none__"][1.0], 0.8263207)
#     assert np.isclose(svy_prop_without_str.coef_var["__none__"][0.0], 0.0066567 / 0.1863775)
#     assert np.isclose(svy_prop_without_str.coef_var["__none__"][1.0], 0.0066567 / 0.8136225)


# svy_prop_without_str_domain = TaylorEstimator("proportion")


# def test_prop_estimator_without_str_domain():
#     svy_prop_without_str_domain.estimate(y, weight, psu=psu, domain=domain, remove_nan=True)

#     assert np.isclose(svy_prop_without_str_domain.point_est[1][0.0], 0.1688402)
#     assert np.isclose(svy_prop_without_str_domain.point_est[1][1.0], 0.8311598)
#     assert np.isclose(svy_prop_without_str_domain.point_est[2][0.0], 0.202774)
#     assert np.isclose(svy_prop_without_str_domain.point_est[2][1.0], 0.797226)
#     assert np.isclose(svy_prop_without_str_domain.point_est["d3"][0.0], 0.1809641)
#     assert np.isclose(svy_prop_without_str_domain.point_est["d3"][1.0], 0.8190359)
#     assert np.isclose(svy_prop_without_str_domain.stderror[1][0.0], 0.0203778)
#     assert np.isclose(svy_prop_without_str_domain.stderror[1][1.0], 0.0203778)
#     assert np.isclose(svy_prop_without_str_domain.stderror[2][0.0], 0.0260659)
#     assert np.isclose(svy_prop_without_str_domain.stderror[2][1.0], 0.0260659)
#     assert np.isclose(svy_prop_without_str_domain.stderror["d3"][0.0], 0.0190814)
#     assert np.isclose(svy_prop_without_str_domain.stderror["d3"][1.0], 0.0190814)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci[1][0.0], 0.131771)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci[1][1.0], 0.7862299)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci[2][0.0], 0.155414)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci[2][1.0], 0.7398788)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci["d3"][0.0], 0.1457555)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci["d3"][1.0], 0.7775374)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci[1][0.0], 0.2137701)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci[1][1.0], 0.868229)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci[2][0.0], 0.2601212)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci[2][1.0], 0.844586)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci["d3"][0.0], 0.2224624)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci["d3"][1.0], 0.8542445)


# def test_prop_estimator_without_str_nor_psu_domain():
#     svy_prop_without_str_domain.estimate(y, weight, domain=domain, remove_nan=True)

#     assert np.isclose(svy_prop_without_str_domain.point_est[1][0.0], 0.1688402)
#     assert np.isclose(svy_prop_without_str_domain.point_est[1][1.0], 0.8311598)
#     assert np.isclose(svy_prop_without_str_domain.point_est[2][0.0], 0.202774)
#     assert np.isclose(svy_prop_without_str_domain.point_est[2][1.0], 0.797226)
#     assert np.isclose(svy_prop_without_str_domain.point_est["d3"][0.0], 0.1809641)
#     assert np.isclose(svy_prop_without_str_domain.point_est["d3"][1.0], 0.8190359)
#     assert np.isclose(svy_prop_without_str_domain.stderror[1][0.0], 0.0200196)
#     assert np.isclose(svy_prop_without_str_domain.stderror[1][1.0], 0.0200196)
#     assert np.isclose(svy_prop_without_str_domain.stderror[2][0.0], 0.0125303)
#     assert np.isclose(svy_prop_without_str_domain.stderror[2][1.0], 0.0125303)
#     assert np.isclose(svy_prop_without_str_domain.stderror["d3"][0.0], 0.0085047)
#     assert np.isclose(svy_prop_without_str_domain.stderror["d3"][1.0], 0.0085047)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci[1][0.0], 0.133136)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci[1][1.0], 0.7882197)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci[2][0.0], 0.179316)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci[2][1.0], 0.7715536)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci["d3"][0.0], 0.1648868)
#     assert np.isclose(svy_prop_without_str_domain.lower_ci["d3"][1.0], 0.8017632)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci[1][0.0], 0.2117803)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci[1][1.0], 0.866864)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci[2][0.0], 0.2284464)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci[2][1.0], 0.820684)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci["d3"][0.0], 0.1982368)
#     assert np.isclose(svy_prop_without_str_domain.upper_ci["d3"][1.0], 0.8351132)


# """Taylor Approximation WITH Stratification for PROPORTION"""
# svy_prop_with_str = TaylorEstimator("proportion")


# def test_prop_estimator_with_str():
#     svy_prop_with_str.estimate(y, weight, stratum=stratum, psu=psu, remove_nan=True)

#     assert np.isclose(svy_prop_with_str.point_est["__none__"][0.0], 0.186_377_5)
#     assert np.isclose(svy_prop_with_str.point_est["__none__"][1.0], 0.813_622_5)
#     assert np.isclose(svy_prop_with_str.variance["__none__"][0.0], 0.019_862_1 ** 2)
#     assert np.isclose(svy_prop_with_str.variance["__none__"][1.0], 0.019_862_1 ** 2)
#     assert np.isclose(svy_prop_with_str.stderror["__none__"][0.0], 0.019_862_1)
#     assert np.isclose(svy_prop_with_str.stderror["__none__"][1.0], 0.019_862_1)
#     assert np.isclose(svy_prop_with_str.lower_ci["__none__"][0.0], 0.149_483_7)
#     assert np.isclose(svy_prop_with_str.lower_ci["__none__"][1.0], 0.770_084_5)
#     assert np.isclose(svy_prop_with_str.upper_ci["__none__"][0.0], 0.229_915_5)
#     assert np.isclose(svy_prop_with_str.upper_ci["__none__"][1.0], 0.850_516_3)
#     assert np.isclose(svy_prop_with_str.coef_var["__none__"][0.0], 0.019_862_1 / 0.186_377_5)
#     assert np.isclose(svy_prop_with_str.coef_var["__none__"][1.0], 0.019_862_1 / 0.813_622_5)


# def test_prop_estimator_with_str_without_psu():
#     svy_prop_with_str.estimate(y, weight, stratum=stratum, remove_nan=True)

#     assert np.isclose(svy_prop_with_str.point_est["__none__"][0.0], 0.1863775)
#     assert np.isclose(svy_prop_with_str.point_est["__none__"][1.0], 0.8136225)
#     assert np.isclose(svy_prop_with_str.variance["__none__"][0.0], 0.0066091 ** 2)
#     assert np.isclose(svy_prop_with_str.variance["__none__"][1.0], 0.0066091 ** 2)
#     assert np.isclose(svy_prop_with_str.stderror["__none__"][0.0], 0.0066091)
#     assert np.isclose(svy_prop_with_str.stderror["__none__"][1.0], 0.0066091)
#     assert np.isclose(svy_prop_with_str.lower_ci["__none__"][0.0], 0.1737677)
#     assert np.isclose(svy_prop_with_str.lower_ci["__none__"][1.0], 0.8003188)
#     assert np.isclose(svy_prop_with_str.upper_ci["__none__"][0.0], 0.1996812)
#     assert np.isclose(svy_prop_with_str.upper_ci["__none__"][1.0], 0.8262323)
#     assert np.isclose(svy_prop_with_str.coef_var["__none__"][0.0], 0.0066091 / 0.1863775)
#     assert np.isclose(svy_prop_with_str.coef_var["__none__"][1.0], 0.0066091 / 0.8136225)


# svy_prop_with_str_domain = TaylorEstimator("proportion")


# def test_prop_estimator_with_str_domain():
#     svy_prop_with_str_domain.estimate(
#         y, weight, psu=psu, stratum=stratum, domain=domain, remove_nan=True
#     )

#     assert np.isclose(svy_prop_with_str_domain.point_est[1][0.0], 0.1688402)
#     assert np.isclose(svy_prop_with_str_domain.point_est[1][1.0], 0.8311598)
#     assert np.isclose(svy_prop_with_str_domain.point_est[2][0.0], 0.202774)
#     assert np.isclose(svy_prop_with_str_domain.point_est[2][1.0], 0.797226)
#     assert np.isclose(svy_prop_with_str_domain.point_est["d3"][0.0], 0.1809641)
#     assert np.isclose(svy_prop_with_str_domain.point_est["d3"][1.0], 0.8190359)
#     assert np.isclose(svy_prop_with_str_domain.stderror[1][0.0], 0.0200457)
#     assert np.isclose(svy_prop_with_str_domain.stderror[1][1.0], 0.0200457)
#     assert np.isclose(svy_prop_with_str_domain.stderror[2][0.0], 0.0263015)
#     assert np.isclose(svy_prop_with_str_domain.stderror[2][1.0], 0.0263015)
#     assert np.isclose(svy_prop_with_str_domain.stderror["d3"][0.0], 0.0182081)
#     assert np.isclose(svy_prop_with_str_domain.stderror["d3"][1.0], 0.0182081)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci[1][0.0], 0.1320679)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci[1][1.0], 0.7866654)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci[2][0.0], 0.1547087)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci[2][1.0], 0.7388414)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci["d3"][0.0], 0.1470016)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci["d3"][1.0], 0.7792576)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci[1][0.0], 0.2133346)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci[1][1.0], 0.8679321)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci[2][0.0], 0.2611586)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci[2][1.0], 0.8452913)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci["d3"][0.0], 0.2207424)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci["d3"][1.0], 0.8529984)


# def test_prop_estimator_with_str_nor_psu_domain():
#     svy_prop_with_str_domain.estimate(y, weight, stratum=stratum, domain=domain, remove_nan=True)

#     assert np.isclose(svy_prop_with_str_domain.point_est[1][0.0], 0.1688402)
#     assert np.isclose(svy_prop_with_str_domain.point_est[1][1.0], 0.8311598)
#     assert np.isclose(svy_prop_with_str_domain.point_est[2][0.0], 0.202774)
#     assert np.isclose(svy_prop_with_str_domain.point_est[2][1.0], 0.797226)
#     assert np.isclose(svy_prop_with_str_domain.point_est["d3"][0.0], 0.1809641)
#     assert np.isclose(svy_prop_with_str_domain.point_est["d3"][1.0], 0.8190359)
#     assert np.isclose(svy_prop_with_str_domain.stderror[1][0.0], 0.0200198)
#     assert np.isclose(svy_prop_with_str_domain.stderror[1][1.0], 0.0200198)
#     assert np.isclose(svy_prop_with_str_domain.stderror[2][0.0], 0.0125144)
#     assert np.isclose(svy_prop_with_str_domain.stderror[2][1.0], 0.0125144)
#     assert np.isclose(svy_prop_with_str_domain.stderror["d3"][0.0], 0.0084659)
#     assert np.isclose(svy_prop_with_str_domain.stderror["d3"][1.0], 0.0084659)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci[1][0.0], 0.1331356)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci[1][1.0], 0.7882192)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci[2][0.0], 0.1793444)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci[2][1.0], 0.7715876)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci["d3"][0.0], 0.1649573)
#     assert np.isclose(svy_prop_with_str_domain.lower_ci["d3"][1.0], 0.8018446)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci[1][0.0], 0.2117808)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci[1][1.0], 0.8668644)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci[2][0.0], 0.2284124)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci[2][1.0], 0.8206556)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci["d3"][0.0], 0.1981554)
#     assert np.isclose(svy_prop_with_str_domain.upper_ci["d3"][1.0], 0.8350427)
