import numpy as np
import pandas as pd
import pytest

from samplics.estimation import TaylorEstimator
from samplics.utils.types import PopParam, SinglePSUEst


sample_data1 = pd.DataFrame.from_dict(
    data={
        "region": [1, 1, 1, 2, 2, 3, 3, 4],
        "district": [1, 2, 2, 1, 2, 1, 2, 1],
        "area": [1, 1, 2, 1, 2, 1, 1, 1],
        "domain": [1, 1, 1, 1, 1, 2, 2, 2],
        "wgt": [1.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5],
        "age": [12, 34, 24, 12, 33, 46, 78, 98],
    }
)


def test_single_psu_mean_skip():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        single_psu=SinglePSUEst.skip,
        remove_nan=True,
    )
    assert np.isclose(svy_mean_single_psu.point_est, 52.0238095)
    assert np.isclose(svy_mean_single_psu.variance, 34.879578532)
    assert np.isclose(svy_mean_single_psu.stderror, 5.905893542)
    assert np.isclose(svy_mean_single_psu.lower_ci, 33.22862044)
    assert np.isclose(svy_mean_single_psu.upper_ci, 70.8189986066)
    assert np.isclose(svy_mean_single_psu.coef_var, 5.905893542 / 52.0238095)


def test_single_psu_mean_domain_skip():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        domain=sample_data1["domain"],
        single_psu=SinglePSUEst.skip,
        remove_nan=True,
    )

    assert np.isclose(svy_mean_single_psu.point_est[1], 22.8947368)
    assert np.isclose(svy_mean_single_psu.point_est[2], 76.0869565)
    assert np.isclose(svy_mean_single_psu.variance[1], 43.84952540265)
    assert np.isclose(svy_mean_single_psu.variance[2], 94.85066162570)
    assert np.isclose(svy_mean_single_psu.stderror[1], 6.62189741710)
    assert np.isclose(svy_mean_single_psu.stderror[2], 9.7391304347)
    assert np.isclose(svy_mean_single_psu.lower_ci[1], 1.8209038730)
    assert np.isclose(svy_mean_single_psu.lower_ci[2], 45.092696852)
    assert np.isclose(svy_mean_single_psu.upper_ci[1], 43.968569811)
    assert np.isclose(svy_mean_single_psu.upper_ci[2], 107.081216190)
    assert np.isclose(svy_mean_single_psu.coef_var[1], 6.62189741710 / 22.8947368)
    assert np.isclose(svy_mean_single_psu.coef_var[2], 9.7391304347 / 76.0869565)


@pytest.mark.xfail
def test_single_psu_mean_certainty11():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        single_psu=SinglePSUEst.certainty,
        remove_nan=True,
    )


def test_single_psu_mean_certainty12():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=[1, 1, 1, 2, 2, 3, 3, 3],
        psu=[1, 2, 2, 1, 2, 1, 1, 1],
        single_psu=SinglePSUEst.certainty,
        remove_nan=True,
    )


@pytest.mark.xfail
def test_single_psu_mean_domain_certainty11():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        domain=sample_data1["domain"],
        single_psu=SinglePSUEst.certainty,
        remove_nan=True,
    )


def test_single_psu_mean_domain_certainty12():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=[1, 1, 1, 2, 2, 3, 3, 3],
        psu=[1, 2, 2, 1, 2, 1, 1, 1],
        domain=sample_data1["domain"],
        single_psu=SinglePSUEst.certainty,
        remove_nan=True,
    )


@pytest.mark.xfail
def test_single_psu_mean_domain_certainty21():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        ssu=sample_data1["area"],
        domain=sample_data1["domain"],
        single_psu=SinglePSUEst.certainty,
        remove_nan=True,
    )


def test_single_psu_mean_domain_certainty22():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=[1, 1, 1, 2, 2, 3, 3, 3],
        psu=[1, 2, 2, 1, 2, 1, 1, 1],
        ssu=[1, 1, 2, 1, 2, 1, 1, 2],
        domain=sample_data1["domain"],
        single_psu=SinglePSUEst.certainty,
        remove_nan=True,
    )


def test_single_psu_mean_combine11():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        single_psu=SinglePSUEst.combine,
        strata_comb={4: 3},
        remove_nan=True,
    )


def test_single_psu_mean_combine12():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data1["age"],
        samp_weight=sample_data1["wgt"],
        stratum=sample_data1["region"],
        psu=sample_data1["district"],
        single_psu=SinglePSUEst.combine,
        strata_comb={4: 2, 3: 2},
        remove_nan=True,
    )


sample_data2 = pd.DataFrame.from_dict(
    data={
        "region": [1, 1, 1, 2, 2, 3, 3, 4],
        "district": [1, 2, 1, 1, 1, 1, 1, 1],
        "area": [1, 1, 2, 1, 2, 1, 1, 1],
        "domain": [1, 1, 1, 1, 1, 2, 2, 2],
        "wgt": [1.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5],
        "age": [12, 34, 24, 12, 33, 46, 78, 98],
    }
)


def test_single_psu_mean_dict_skip():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data2["age"],
        samp_weight=sample_data2["wgt"],
        stratum=sample_data2["region"],
        psu=sample_data2["district"],
        single_psu={(2, 3, 4): SinglePSUEst.skip},
        remove_nan=True,
    )
    assert np.isclose(svy_mean_single_psu.point_est, 52.0238095)
    assert np.isclose(svy_mean_single_psu.variance, 12.76725265)
    assert np.isclose(svy_mean_single_psu.stderror, 3.573129251)
    assert np.isclose(svy_mean_single_psu.lower_ci, 6.622897701)
    assert np.isclose(svy_mean_single_psu.upper_ci, 97.424721345)
    assert np.isclose(svy_mean_single_psu.coef_var, 3.573129251 / 52.0238095)


def test_single_psu_mean_domain_dict_skip():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data2["age"],
        samp_weight=sample_data2["wgt"],
        stratum=sample_data2["region"],
        psu=sample_data2["district"],
        domain=sample_data2["domain"],
        single_psu={(2, 3, 4): SinglePSUEst.skip},
        remove_nan=True,
    )

    assert np.isclose(svy_mean_single_psu.point_est[1], 22.8947368)
    assert np.isclose(svy_mean_single_psu.point_est[2], 76.0869565)
    assert np.isclose(svy_mean_single_psu.variance[1], 10.88451592)
    assert np.isclose(svy_mean_single_psu.variance[2], 0.0)
    assert np.isclose(svy_mean_single_psu.stderror[1], 3.299168975)
    assert np.isclose(svy_mean_single_psu.stderror[2], 0.0)
    assert np.isclose(svy_mean_single_psu.lower_ci[1], -19.0251796)
    assert np.isclose(svy_mean_single_psu.lower_ci[2], 76.0869565)
    assert np.isclose(svy_mean_single_psu.upper_ci[1], 64.814653)
    assert np.isclose(svy_mean_single_psu.lower_ci[2], svy_mean_single_psu.upper_ci[2])
    assert np.isclose(svy_mean_single_psu.coef_var[1], 3.299168975 / 22.8947368)
    assert np.isclose(svy_mean_single_psu.coef_var[2], 0.0)


@pytest.mark.xfail
def test_single_psu_mean_dict_certainty11():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data2["age"],
        samp_weight=sample_data2["wgt"],
        stratum=sample_data2["region"],
        psu=sample_data2["district"],
        single_psu={(2, 3, 4): SinglePSUEst.certainty},
        remove_nan=True,
    )


def test_single_psu_mean_dict_certainty12():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data2["age"],
        samp_weight=sample_data2["wgt"],
        stratum=[1, 1, 1, 2, 2, 3, 3, 3],
        psu=sample_data2["district"],
        single_psu={(2, 3, 4): SinglePSUEst.certainty},
        remove_nan=True,
    )


def test_single_psu_mean_domain_dict_certainty11():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data2["age"],
        samp_weight=sample_data2["wgt"],
        stratum=sample_data2["region"],
        psu=sample_data2["district"],
        domain=sample_data2["domain"],
        single_psu={(2, 3): SinglePSUEst.certainty, 4: SinglePSUEst.skip},
        remove_nan=True,
    )


@pytest.mark.xfail
def test_single_psu_mean_domain_dict_certainty12():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data2["age"],
        samp_weight=sample_data2["wgt"],
        stratum=sample_data2["region"],
        psu=sample_data2["district"],
        ssu=sample_data2["area"],
        domain=sample_data2["domain"],
        single_psu={(2, 3): SinglePSUEst.certainty, 4: SinglePSUEst.skip},
        remove_nan=True,
    )


def test_single_psu_mean_domain_dict_certainty13():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data2["age"],
        samp_weight=sample_data2["wgt"],
        stratum=sample_data2["region"],
        psu=sample_data2["district"],
        ssu=[1, 2, 1, 2, 1, 1, 2, 1],
        domain=sample_data2["domain"],
        single_psu={(2, 3): SinglePSUEst.certainty, 4: SinglePSUEst.skip},
        remove_nan=True,
    )


def test_single_psu_mean_dict_combine11():
    svy_mean_single_psu = TaylorEstimator(param=PopParam.mean)
    svy_mean_single_psu.estimate(
        y=sample_data2["age"],
        samp_weight=sample_data2["wgt"],
        stratum=sample_data2["region"],
        psu=sample_data2["district"],
        single_psu={(2, 3): SinglePSUEst.certainty, 4: SinglePSUEst.skip},
        strata_comb={2: 1, 3: 1},
        remove_nan=True,
    )
