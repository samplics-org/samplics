import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator

from samplics.utils.types import SinglePSUEst

sample_data = pd.DataFrame.from_dict(
    data={
        "region": [1, 1, 1, 2, 2, 3, 3, 4],
        "district": [1, 2, 1, 1, 1, 1, 1, 1],
        "area": [1, 1, 2, 1, 2, 1, 1, 1],
        "domain": [1, 1, 1, 1, 1, 2, 2, 2],
        "wgt": [1.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5],
        "age": [12, 34, 24, 12, 33, 46, 78, 98],
    }
)

svy_mean_single_psu1 = TaylorEstimator(parameter="mean")


def test_single_psu_total():
    svy_mean_single_psu1.estimate(
        y=sample_data["age"],
        samp_weight=sample_data["wgt"],
        stratum=sample_data["region"],
        psu=sample_data["district"],
        single_psu=SinglePSUEst.skip,
        remove_nan=True,
    )
    assert np.isclose(svy_mean_single_psu1.point_est, 52.0238095)
    assert np.isclose(svy_mean_single_psu1.variance, 12.76725265)
    assert np.isclose(svy_mean_single_psu1.stderror, 3.573129251)
    assert np.isclose(svy_mean_single_psu1.lower_ci, 6.622897701)
    assert np.isclose(svy_mean_single_psu1.upper_ci, 97.424721345)
    assert np.isclose(svy_mean_single_psu1.coef_var, 3.573129251 / 52.0238095)


svy_mean_single_psu2 = TaylorEstimator(parameter="mean")


def test_single_psu_mean_domain_skip():
    svy_mean_single_psu2.estimate(
        y=sample_data["age"],
        samp_weight=sample_data["wgt"],
        stratum=sample_data["region"],
        psu=sample_data["district"],
        domain=sample_data["domain"],
        single_psu=SinglePSUEst.skip,
        remove_nan=True,
    )

    assert np.isclose(svy_mean_single_psu2.point_est[1], 22.8947368)
    assert np.isclose(svy_mean_single_psu2.point_est[2], 76.0869565)
    assert np.isclose(svy_mean_single_psu2.variance[1], 10.88451592)
    assert np.isclose(svy_mean_single_psu2.variance[2], 0.0)
    assert np.isclose(svy_mean_single_psu2.stderror[1], 3.299168975)
    assert np.isclose(svy_mean_single_psu2.stderror[2], 0.0)
    assert np.isclose(svy_mean_single_psu2.lower_ci[1], -19.0251796)
    assert np.isclose(svy_mean_single_psu2.lower_ci[2], 76.0869565)
    assert np.isclose(svy_mean_single_psu2.upper_ci[1], 64.814653)
    assert np.isclose(svy_mean_single_psu2.lower_ci[2], svy_mean_single_psu2.upper_ci[2])
    assert np.isclose(svy_mean_single_psu2.coef_var[1], 3.299168975 / 22.8947368)
    assert np.isclose(svy_mean_single_psu2.coef_var[2], 0.0)
