import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator

from samplics.utils.types import SinglePSUEst

np.random.seed(12345)

yrbs = pd.read_csv("./tests/estimation/yrbs.csv")

yrbs["y"] = yrbs["qn8"].replace({2: 0})
yrbs["x"] = 0.8 * yrbs["y"] + 0.5
yrbs["domain"] = np.random.choice(["d1", "d2", "d3"], size=yrbs.shape[0], p=[0.1, 0.3, 0.6])
yrbs["by"] = np.random.choice(["b1", "b2"], size=yrbs.shape[0], p=[0.4, 0.6])

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

"""Taylor Approximation WITH Stratification for TOTAL"""
svy_total_with_str = TaylorEstimator("total")


def test_total_estimator_with_str():
    svy_total_with_str.estimate(y, weight, stratum=stratum, psu=psu, remove_nan=True)

    # assert np.isclose(svy_total_with_str.point_est, 7938.333)
    # assert np.isclose(svy_total_with_str.variance, 555.5157**2)
    # assert np.isclose(svy_total_with_str.stderror, 555.5157)
    # assert np.isclose(svy_total_with_str.lower_ci, 6814.697)
    # assert np.isclose(svy_total_with_str.upper_ci, 9061.970)
    # assert np.isclose(svy_total_with_str.coef_var, 555.5157 / 7938.333)


# sample_data = pd.DataFrame.from_dict(
#     data={
#         "region": [1, 1, 1, 2, 2, 3, 3, 3],
#         "district": [1, 1, 1, 1, 1, 1, 1, 1],
#         "area": [1, 1, 2, 1, 2, 1, 1, 1],
#         "domain": [1, 1, 1, 1, 1, 2, 2, 2],
#         "wgt": [1.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5],
#         "age": [12, 34, 24, 12, 33, 46, 78, 98],
#     }
# )


# def test_single_psu_mean_skip():
#     svy_mean_single_psu = TaylorEstimator(parameter="mean")
#     svy_mean_single_psu.estimate(
#         y=sample_data["age"],
#         samp_weight=sample_data["wgt"],
#         stratum=sample_data["region"],
#         psu=sample_data["district"],
#         single_psu=SinglePSUEst.skip,
#         remove_nan=True,
#     )
#     assert np.isclose(svy_mean_single_psu.point_est, 52.0238095)
#     assert np.isclose(svy_mean_single_psu.variance, 12.76725265)
#     assert np.isclose(svy_mean_single_psu.stderror, 3.573129251)
#     assert np.isclose(svy_mean_single_psu.lower_ci, 6.622897701)
#     assert np.isclose(svy_mean_single_psu.upper_ci, 97.424721345)
#     assert np.isclose(svy_mean_single_psu.coef_var, 3.573129251 / 52.0238095)


# def test_single_psu_mean_domain_skip():
#     svy_mean_single_psu = TaylorEstimator(parameter="mean")
#     svy_mean_single_psu.estimate(
#         y=sample_data["age"],
#         samp_weight=sample_data["wgt"],
#         stratum=sample_data["region"],
#         psu=sample_data["district"],
#         domain=sample_data["domain"],
#         single_psu=SinglePSUEst.skip,
#         remove_nan=True,
#     )

#     assert np.isclose(svy_mean_single_psu.point_est[1], 22.8947368)
#     assert np.isclose(svy_mean_single_psu.point_est[2], 76.0869565)
#     assert np.isclose(svy_mean_single_psu.variance[1], 10.88451592)
#     assert np.isclose(svy_mean_single_psu.variance[2], 0.0)
#     assert np.isclose(svy_mean_single_psu.stderror[1], 3.299168975)
#     assert np.isclose(svy_mean_single_psu.stderror[2], 0.0)
#     assert np.isclose(svy_mean_single_psu.lower_ci[1], -19.0251796)
#     assert np.isclose(svy_mean_single_psu.lower_ci[2], 76.0869565)
#     assert np.isclose(svy_mean_single_psu.upper_ci[1], 64.814653)
#     assert np.isclose(svy_mean_single_psu.lower_ci[2], svy_mean_single_psu.upper_ci[2])
#     assert np.isclose(svy_mean_single_psu.coef_var[1], 3.299168975 / 22.8947368)
#     assert np.isclose(svy_mean_single_psu.coef_var[2], 0.0)


# def test_single_psu_mean_certainty():
#     svy_mean_single_psu = TaylorEstimator(parameter="mean")
#     svy_mean_single_psu.estimate(
#         y=sample_data["age"],
#         samp_weight=sample_data["wgt"],
#         stratum=sample_data["region"],
#         psu=sample_data["district"],
#         single_psu=SinglePSUEst.certainty,
#         remove_nan=True,
#     )


# def test_single_psu_mean_domain_certainty1():
#     svy_mean_single_psu = TaylorEstimator(parameter="mean")
#     svy_mean_single_psu.estimate(
#         y=sample_data["age"],
#         samp_weight=sample_data["wgt"],
#         stratum=sample_data["region"],
#         psu=sample_data["district"],
#         domain=sample_data["domain"],
#         single_psu=SinglePSUEst.certainty,
#         remove_nan=True,
#     )


# def test_single_psu_mean_domain_certainty2():
#     svy_mean_single_psu = TaylorEstimator(parameter="mean")
#     svy_mean_single_psu.estimate(
#         y=sample_data["age"],
#         samp_weight=sample_data["wgt"],
#         stratum=sample_data["region"],
#         psu=sample_data["district"],
#         ssu=sample_data["area"],
#         domain=sample_data["domain"],
#         single_psu=SinglePSUEst.certainty,
#         remove_nan=True,
#     )


# def test_single_psu_mean_combine():
#     svy_mean_single_psu = TaylorEstimator(parameter="mean")
#     svy_mean_single_psu.estimate(
#         y=sample_data["age"],
#         samp_weight=sample_data["wgt"],
#         stratum=sample_data["region"],
#         psu=sample_data["district"],
#         # single_psu=SinglePSUEst.combine,
#         # strata_comb={4: 3},
#         remove_nan=True,
#     )


#     breakpoint()
