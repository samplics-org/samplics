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
z = one_stage_fpc["z"]
fpc_psu = 1 - 5 / 44
fpc_no_psu = 1 - 21 / 44

"""Taylor Approximation WITHOUT Stratification for TOTAL"""
svy_total_without_str = TaylorEstimator("total")


def test_total_estimator_without_str():
    svy_total_without_str.estimate(y, weight, psu=psu, fpc=fpc_psu)

    assert np.isclose(svy_total_without_str.point_est, 853.12)
    assert np.isclose(svy_total_without_str.stderror, 153.2627)
    assert np.isclose(svy_total_without_str.lower_ci, 427.5946)
    assert np.isclose(svy_total_without_str.upper_ci, 1278.645)


def test_total_estimator_without_str_nor_psu():
    svy_total_without_str.estimate(y, weight, fpc=fpc_no_psu)

    assert np.isclose(svy_total_without_str.point_est, 853.12)
    assert np.isclose(svy_total_without_str.stderror, 67.14785)
    assert np.isclose(svy_total_without_str.lower_ci, 713.052)
    assert np.isclose(svy_total_without_str.upper_ci, 993.188)


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

    assert np.isclose(svy_total_with_str.point_est, 853.12)
    assert np.isclose(svy_total_with_str.stderror, 97.41386)
    assert np.isclose(svy_total_with_str.lower_ci, 632.75452)
    assert np.isclose(svy_total_with_str.upper_ci, 1073.48547)


def test_total_estimator_with_str_without_psu():
    svy_total_with_str.estimate(y, weight, stratum=stratum, fpc=fpc_dict_no_psu)

    assert np.isclose(svy_total_with_str.point_est, 853.12)
    assert np.isclose(svy_total_with_str.stderror, 50.8888)
    assert np.isclose(svy_total_with_str.lower_ci, 746.2066)
    assert np.isclose(svy_total_with_str.upper_ci, 960.0334)


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

    assert np.isclose(svy_mean_without_str.point_est, 10.27855)
    assert np.isclose(svy_mean_without_str.stderror, 1.126881)
    assert np.isclose(svy_mean_without_str.lower_ci, 7.14983)
    assert np.isclose(svy_mean_without_str.upper_ci, 13.40728)


def test_mean_estimator_without_str_nor_psu():
    svy_mean_without_str.estimate(y, weight, fpc=fpc_no_psu)

    assert np.isclose(svy_mean_without_str.point_est, 10.27855)
    assert np.isclose(svy_mean_without_str.stderror, 0.6177327)
    assert np.isclose(svy_mean_without_str.lower_ci, 8.989986)
    assert np.isclose(svy_mean_without_str.upper_ci, 11.56712)


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

    assert np.isclose(svy_mean_with_str.point_est, 10.27855)
    assert np.isclose(svy_mean_with_str.stderror, 0.7750964)
    assert np.isclose(svy_mean_with_str.lower_ci, 8.525164)
    assert np.isclose(svy_mean_with_str.upper_ci, 12.03194)


def test_mean_estimator_with_str_without_psu():
    svy_mean_with_str.estimate(y, weight, stratum=stratum, fpc=fpc_dict_no_psu)

    assert np.isclose(svy_mean_with_str.point_est, 10.27855)
    assert np.isclose(svy_mean_with_str.stderror, 0.6131181)
    assert np.isclose(svy_mean_with_str.lower_ci, 8.990441)
    assert np.isclose(svy_mean_with_str.upper_ci, 11.56667)


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

    assert np.isclose(svy_ratio_without_str.point_est, 1.892038)
    assert np.isclose(svy_ratio_without_str.stderror, 0.0106065)
    assert np.isclose(svy_ratio_without_str.lower_ci, 1.86259)
    assert np.isclose(svy_ratio_without_str.upper_ci, 1.921486)


def test_ratio_estimator_without_str_nor_psu():
    svy_ratio_without_str.estimate(y, weight, x, fpc=fpc_no_psu)

    assert np.isclose(svy_ratio_without_str.point_est, 1.892038)
    assert np.isclose(svy_ratio_without_str.stderror, 0.0058142)
    assert np.isclose(svy_ratio_without_str.lower_ci, 1.87991)
    assert np.isclose(svy_ratio_without_str.upper_ci, 1.904166)


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

    assert np.isclose(svy_ratio_with_str.point_est, 1.892038)
    assert np.isclose(svy_ratio_with_str.stderror, 0.0072954)
    assert np.isclose(svy_ratio_with_str.lower_ci, 1.875535)
    assert np.isclose(svy_ratio_with_str.upper_ci, 1.908542)


def test_ratio_estimator_with_str_without_psu():
    svy_ratio_with_str.estimate(y, weight, x, stratum=stratum, fpc=fpc_dict_no_psu)

    assert np.isclose(svy_ratio_with_str.point_est, 1.892038)
    assert np.isclose(svy_ratio_with_str.stderror, 0.0057708)
    assert np.isclose(svy_ratio_with_str.lower_ci, 1.879914)
    assert np.isclose(svy_ratio_with_str.upper_ci, 1.904162)


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


"""Taylor Approximation WITHOUT Stratification for PROPORTION"""
svy_prop_without_str = TaylorEstimator("proportion")


def test_prop_estimator_without_str():
    svy_prop_without_str.estimate(z, weight, psu=psu, fpc=fpc_psu)

    assert np.isclose(svy_prop_without_str.point_est[0.0], 0.3253012)
    assert np.isclose(svy_prop_without_str.point_est[1.0], 0.6746988)
    assert np.isclose(svy_prop_without_str.stderror[0.0], 0.1499394)
    assert np.isclose(svy_prop_without_str.stderror[1.0], 0.1499394)
    assert np.isclose(svy_prop_without_str.lower_ci[0.0], 0.0674673)
    assert np.isclose(svy_prop_without_str.lower_ci[1.0], 0.2373558)
    assert np.isclose(svy_prop_without_str.upper_ci[0.0], 0.7626442)
    assert np.isclose(svy_prop_without_str.upper_ci[1.0], 0.9325327)


def test_prop_estimator_without_str_nor_psu():
    svy_prop_without_str.estimate(z, weight, fpc=fpc_no_psu)

    assert np.isclose(svy_prop_without_str.point_est[0.0], 0.3253012)
    assert np.isclose(svy_prop_without_str.point_est[1.0], 0.6746988)
    assert np.isclose(svy_prop_without_str.stderror[0.0], 0.0776683)
    assert np.isclose(svy_prop_without_str.stderror[1.0], 0.0776683)
    assert np.isclose(svy_prop_without_str.lower_ci[0.0], 0.1872951)
    assert np.isclose(svy_prop_without_str.lower_ci[1.0], 0.4978368)
    assert np.isclose(svy_prop_without_str.upper_ci[0.0], 0.5021632)
    assert np.isclose(svy_prop_without_str.upper_ci[1.0], 0.8127049)


svy_prop_without_str_domain = TaylorEstimator("proportion")


def test_prop_estimator_without_str_domain():
    svy_prop_without_str_domain.estimate(z, weight, psu=psu, domain=domain, fpc=fpc_psu)

    assert np.isclose(svy_prop_without_str_domain.point_est[1][0.0], 0.1132075)
    assert np.isclose(svy_prop_without_str_domain.point_est[1][1.0], 0.8867925)
    assert np.isclose(svy_prop_without_str_domain.point_est[2][0.0], 0.7)
    assert np.isclose(svy_prop_without_str_domain.point_est[2][1.0], 0.3)
    assert np.isclose(svy_prop_without_str_domain.stderror[1][0.0], 0.1140682)
    assert np.isclose(svy_prop_without_str_domain.stderror[1][1.0], 0.1140682)
    assert np.isclose(svy_prop_without_str_domain.stderror[2][0.0], 0.1042015)
    assert np.isclose(svy_prop_without_str_domain.stderror[2][1.0], 0.1042015)
    assert np.isclose(svy_prop_without_str_domain.lower_ci[1][0.0], 0.0054154)
    assert np.isclose(svy_prop_without_str_domain.lower_ci[1][1.0], 0.2504337)
    assert np.isclose(svy_prop_without_str_domain.lower_ci[2][0.0], 0.3704311)
    assert np.isclose(svy_prop_without_str_domain.lower_ci[2][1.0], 0.0975311)
    assert np.isclose(svy_prop_without_str_domain.upper_ci[1][0.0], 0.7495663)
    assert np.isclose(svy_prop_without_str_domain.upper_ci[1][1.0], 0.9945846)
    assert np.isclose(svy_prop_without_str_domain.upper_ci[2][0.0], 0.9024689)
    assert np.isclose(svy_prop_without_str_domain.upper_ci[2][1.0], 0.6295689)


def test_prop_estimator_without_str_nor_psu_domain():
    svy_prop_without_str_domain.estimate(z, weight, domain=domain, fpc=fpc_no_psu)

    assert np.isclose(svy_prop_without_str_domain.point_est[1][0.0], 0.1132075)
    assert np.isclose(svy_prop_without_str_domain.point_est[1][1.0], 0.8867925)
    assert np.isclose(svy_prop_without_str_domain.point_est[2][0.0], 0.7)
    assert np.isclose(svy_prop_without_str_domain.point_est[2][1.0], 0.3)
    assert np.isclose(svy_prop_without_str_domain.stderror[1][0.0], 0.0573954)
    assert np.isclose(svy_prop_without_str_domain.stderror[1][1.0], 0.0573954)
    assert np.isclose(svy_prop_without_str_domain.stderror[2][0.0], 0.1147725)
    assert np.isclose(svy_prop_without_str_domain.stderror[2][1.0], 0.1147725)
    assert np.isclose(svy_prop_without_str_domain.lower_ci[1][0.0], 0.0372922)
    assert np.isclose(svy_prop_without_str_domain.lower_ci[1][1.0], 0.7038733)
    assert np.isclose(svy_prop_without_str_domain.lower_ci[2][0.0], 0.4273294)
    assert np.isclose(svy_prop_without_str_domain.lower_ci[2][1.0], 0.1205374)
    assert np.isclose(svy_prop_without_str_domain.upper_ci[1][0.0], 0.2961267)
    assert np.isclose(svy_prop_without_str_domain.upper_ci[1][1.0], 0.9627078)
    assert np.isclose(svy_prop_without_str_domain.upper_ci[2][0.0], 0.8794626)
    assert np.isclose(svy_prop_without_str_domain.upper_ci[2][1.0], 0.5726706)


"""Taylor Approximation WITH Stratification for PROPORTION"""
svy_prop_with_str = TaylorEstimator("proportion")


def test_prop_estimator_with_str():
    svy_prop_with_str.estimate(z, weight, stratum=stratum, psu=psu, fpc=fpc_dict_psu)

    assert np.isclose(svy_prop_with_str.point_est[0.0], 0.3253012)
    assert np.isclose(svy_prop_with_str.point_est[1.0], 0.6746988)
    assert np.isclose(svy_prop_with_str.stderror[0.0], 0.130421)
    assert np.isclose(svy_prop_with_str.stderror[1.0], 0.130421)
    assert np.isclose(svy_prop_with_str.lower_ci[0.0], 0.1116747)
    assert np.isclose(svy_prop_with_str.lower_ci[1.0], 0.3509837)
    assert np.isclose(svy_prop_with_str.upper_ci[0.0], 0.6490163)
    assert np.isclose(svy_prop_with_str.upper_ci[1.0], 0.8883253)


def test_prop_estimator_with_str_without_psu():
    svy_prop_with_str.estimate(z, weight, stratum=stratum, fpc=fpc_dict_no_psu)

    assert np.isclose(svy_prop_with_str.point_est[0.0], 0.3253012)
    assert np.isclose(svy_prop_with_str.point_est[1.0], 0.6746988)
    assert np.isclose(svy_prop_with_str.stderror[0.0], 0.0789747)
    assert np.isclose(svy_prop_with_str.stderror[1.0], 0.0789747)
    assert np.isclose(svy_prop_with_str.lower_ci[0.0], 0.1846011)
    assert np.isclose(svy_prop_with_str.lower_ci[1.0], 0.4933877)
    assert np.isclose(svy_prop_with_str.upper_ci[0.0], 0.5066123)
    assert np.isclose(svy_prop_with_str.upper_ci[1.0], 0.8153989)


svy_prop_with_str_domain = TaylorEstimator("proportion")


def test_prop_estimator_with_str_domain():
    svy_prop_with_str_domain.estimate(
        z, weight, psu=psu, stratum=stratum, domain=domain, fpc=fpc_dict_psu
    )

    assert np.isclose(svy_prop_with_str_domain.point_est[1][0.0], 0.1132075)
    assert np.isclose(svy_prop_with_str_domain.point_est[1][1.0], 0.8867925)
    assert np.isclose(svy_prop_with_str_domain.point_est[2][0.0], 0.7)
    assert np.isclose(svy_prop_with_str_domain.point_est[2][1.0], 0.3)
    assert np.isclose(svy_prop_with_str_domain.stderror[1][0.0], 0.0705309)
    assert np.isclose(svy_prop_with_str_domain.stderror[1][1.0], 0.0705309)
    assert np.isclose(svy_prop_with_str_domain.stderror[2][0.0], 0.119861)
    assert np.isclose(svy_prop_with_str_domain.stderror[2][1.0], 0.119861)
    assert np.isclose(svy_prop_with_str_domain.lower_ci[1][0.0], 0.02539)
    assert np.isclose(svy_prop_with_str_domain.lower_ci[1][1.0], 0.6151691)
    assert np.isclose(svy_prop_with_str_domain.lower_ci[2][0.0], 0.3908201)
    assert np.isclose(svy_prop_with_str_domain.lower_ci[2][1.0], 0.1054143)
    assert np.isclose(svy_prop_with_str_domain.upper_ci[1][0.0], 0.3848309)
    assert np.isclose(svy_prop_with_str_domain.upper_ci[1][1.0], 0.97461)
    assert np.isclose(svy_prop_with_str_domain.upper_ci[2][0.0], 0.8945857)
    assert np.isclose(svy_prop_with_str_domain.upper_ci[2][1.0], 0.6091799)


def test_prop_estimator_with_str_nor_psu_domain():
    svy_prop_with_str_domain.estimate(
        z, weight, stratum=stratum, domain=domain, fpc=fpc_dict_no_psu
    )

    assert np.isclose(svy_prop_with_str_domain.point_est[1][0.0], 0.1132075)
    assert np.isclose(svy_prop_with_str_domain.point_est[1][1.0], 0.8867925)
    assert np.isclose(svy_prop_with_str_domain.point_est[2][0.0], 0.7)
    assert np.isclose(svy_prop_with_str_domain.point_est[2][1.0], 0.3)
    assert np.isclose(svy_prop_with_str_domain.stderror[1][0.0], 0.0549169)
    assert np.isclose(svy_prop_with_str_domain.stderror[1][1.0], 0.0549169)
    assert np.isclose(svy_prop_with_str_domain.stderror[2][0.0], 0.1046741)
    assert np.isclose(svy_prop_with_str_domain.stderror[2][1.0], 0.1046741)
    assert np.isclose(svy_prop_with_str_domain.lower_ci[1][0.0], 0.0388789)
    assert np.isclose(svy_prop_with_str_domain.lower_ci[1][1.0], 0.7128217)
    assert np.isclose(svy_prop_with_str_domain.lower_ci[2][0.0], 0.4501901)
    assert np.isclose(svy_prop_with_str_domain.lower_ci[2][1.0], 0.1307324)
    assert np.isclose(svy_prop_with_str_domain.upper_ci[1][0.0], 0.2871783)
    assert np.isclose(svy_prop_with_str_domain.upper_ci[1][1.0], 0.9611211)
    assert np.isclose(svy_prop_with_str_domain.upper_ci[2][0.0], 0.8692676)
    assert np.isclose(svy_prop_with_str_domain.upper_ci[2][1.0], 0.5498099)
