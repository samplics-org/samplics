import numpy as np
import polars as pl

from samplics.regression import SurveyGLM
from samplics import ModelType


api_strat = pl.read_csv("./tests/regression/api_strat.csv")

y = api_strat["api00"]
y_bin = (api_strat["api00"].to_numpy() > 743).astype(float)  # api_strat["api00"].quantile(0.75)
x = api_strat.select(["ell", "meals", "mobility"])
x.insert_column(0, pl.Series("intercept", np.ones(x.shape[0])))
stratum = api_strat["stype"]
psu = api_strat["dnum"]
weight = api_strat["pw"]


# Missing data

# def test_reg_logistic_missing():
#     y_bin[20] = np.nan
#     y_bin[35] = np.nan
#     x[33, 1] = np.nan
#     weight[55] = np.nan

#     svyglm = SurveyGLM(model=ModelType.LOGISTIC)
#     svyglm.estimate(y=y_bin, x=x, samp_weight=weight, remove_nan=True)


## Logistic regression


def test_reg_logistic_not_stratified():
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(y=y_bin, x=x, samp_weight=weight)

    assert np.isclose(
        svyglm.beta["point_est"], [2.66575602, -0.0370325, -0.08468788, -0.00442859]
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [0.4718306, 0.03363393, 0.01460186, 0.01478849]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"], [1.74098503, -0.10295378, -0.113307, -0.0334135]
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"], [3.590527, 0.02888879, -0.05606876, 0.02455631]
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [14.37881608, 0.96364482, 0.91879902, 0.9955812],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"], [5.70295826, 0.90216867, 0.89287651, 0.96713857]
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"], [36.25317644, 1.02931011, 0.94547412, 1.0248603]
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 0.4718306**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.0336339**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.0146019**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.0147885**2)


def test_reg_logistic_psu_not_stratified():
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(y=y_bin, x=x, samp_weight=weight, psu=psu)

    assert np.isclose(
        svyglm.beta["point_est"], [2.66575602, -0.0370325, -0.08468788, -0.00442859]
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [0.49495812, 0.0318235, 0.0143854, 0.01537275]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"], [1.69565592, -0.09940542, -0.11288275, -0.03455863]
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"], [3.63585611, 0.02534042, -0.056493, 0.02570144]
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [14.37881608, 0.96364482, 0.91879902, 0.9955812],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"], [5.45021971, 0.90537558, 0.89325539, 0.9660317]
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"], [37.9343151, 1.02566422, 0.9450731, 1.02603457]
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 0.4949581**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.0318235**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.0143854**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.0153727**2)


def test_reg_logistic_stratified():
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(y=y_bin, x=x, samp_weight=weight, stratum=stratum)

    assert np.isclose(
        svyglm.beta["point_est"], [2.66575602, -0.0370325, -0.08468788, -0.00442859]
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [0.43944329, 0.0336371, 0.01433075, 0.01483659]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"], [1.80446299, -0.10296, -0.11277563, -0.03350777]
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"], [3.52704904, 0.028895, -0.05660012, 0.02465058]
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [14.37881608, 0.96364482, 0.91879902, 0.9955812],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"], [6.07670735, 0.90216306, 0.89335108, 0.9670474]
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"], [34.02341764, 1.02931651, 0.94497187, 1.02495692]
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 0.4394433**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.0336371**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.0143307**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.0148366**2)


def test_reg_logistic_psu_stratified():
    svyglm = SurveyGLM(model=ModelType.LOGISTIC)
    svyglm.estimate(
        y=y_bin,
        x=x,
        samp_weight=weight,
        psu=psu,
        stratum=stratum,
    )

    assert np.isclose(
        svyglm.beta["point_est"], [2.66575602, -0.0370325, -0.08468788, -0.00442859]
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [0.47239534, 0.03241909, 0.01428377, 0.015463]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"], [1.73987817, -0.10057275, -0.11268356, -0.03473552]
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"], [3.59163386, 0.02650775, -0.0566922, 0.02587833]
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [14.37881608, 0.96364482, 0.91879902, 0.9955812],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"], [5.69664938, 0.90431932, 0.89343334, 0.96586084]
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"], [36.29332578, 1.02686221, 0.94488486, 1.02621608]
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 0.4723953**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.0324191**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.0142838**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.015463**2)


## Linear regression


def test_reg_linear_not_stratified():
    svyglm = SurveyGLM(model=ModelType.LINEAR)
    svyglm.estimate(y=y, x=x, samp_weight=weight)

    assert np.isclose(
        svyglm.beta["point_est"],
        [8.20887316e02, -4.80586612e-01, -3.14153531e00, 2.25713210e-01],
    ).all()
    assert np.isclose(
        svyglm.beta["stderror"], [10.97090909, 0.39717553, 0.29173325, 0.4012498]
    ).all()
    assert np.isclose(
        svyglm.beta["lower_ci"],
        [7.99384729e02, -1.25903634e00, -3.71332197e00, -5.60721940e-01],
    ).all()
    assert np.isclose(
        svyglm.beta["upper_ci"],
        [8.42389903e02, 2.97863116e-01, -2.56974865e00, 1.01214836e00],
    ).all()

    assert np.isclose(
        svyglm.odds_ratio["point_est"],
        [np.inf, 0.61842051, 0.0432164, 1.2532162],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["lower_ci"],
        [np.inf, 0.2839275, 0.02439634, 0.57079683],
    ).all()
    assert np.isclose(
        svyglm.odds_ratio["upper_ci"],
        [np.inf, 1.3469774, 0.07655479, 2.7515059],
    ).all()

    assert np.isclose(svyglm.beta_cov[0, 0], 10.97091**2)
    assert np.isclose(svyglm.beta_cov[1, 1], 0.3971755**2)
    assert np.isclose(svyglm.beta_cov[2, 2], 0.2917333**2)
    assert np.isclose(svyglm.beta_cov[3, 3], 0.4012498**2)


# def test_reg_linear_psu_not_stratified():
#     svyglm = SurveyGLM(model=ModelType.LINEAR)
#     svyglm.estimate(y=y, x=x, samp_weight=weight, psu=psu)

#     assert np.isclose(svyglm.beta[0], 820.8873)
#     assert np.isclose(svyglm.beta[1], -0.4805866)
#     assert np.isclose(svyglm.beta[2], -3.141535)
#     assert np.isclose(svyglm.beta[3], 0.2257132)

#     assert np.isclose(svyglm.cov_beta[0, 0], 11.65382**2)
#     assert np.isclose(svyglm.cov_beta[1, 1], 0.4292165**2)
#     assert np.isclose(svyglm.cov_beta[2, 2], 0.285711**2)
#     assert np.isclose(svyglm.cov_beta[3, 3], 0.4159033**2)


# def test_reg_linear_stratified():
#     svyglm = SurveyGLM(model=ModelType.LINEAR)
#     svyglm.estimate(y=y, x=x, samp_weight=weight, stratum=stratum)

#     assert np.isclose(svyglm.beta[0], 820.8873)
#     assert np.isclose(svyglm.beta[1], -0.4805866)
#     assert np.isclose(svyglm.beta[2], -3.141535)
#     assert np.isclose(svyglm.beta[3], 0.2257132)

#     assert np.isclose(svyglm.cov_beta[0, 0], 10.25649**2)
#     assert np.isclose(svyglm.cov_beta[1, 1], 0.3977075**2)
#     assert np.isclose(svyglm.cov_beta[2, 2], 0.2883001**2)
#     assert np.isclose(svyglm.cov_beta[3, 3], 0.4026908**2)


# def test_reg_linear_psu_stratified():
#     svyglm = SurveyGLM(model=ModelType.LINEAR)
#     svyglm.estimate(y=y, x=x, samp_weight=weight, psu=psu, stratum=stratum)

#     assert np.isclose(svyglm.beta[0], 820.8873)
#     assert np.isclose(svyglm.beta[1], -0.4805866)
#     assert np.isclose(svyglm.beta[2], -3.141535)
#     assert np.isclose(svyglm.beta[3], 0.2257132)

#     assert np.isclose(svyglm.cov_beta[0, 0], 10.10296**2)
#     assert np.isclose(svyglm.cov_beta[1, 1], 0.4109377**2)
#     assert np.isclose(svyglm.cov_beta[2, 2], 0.2743919**2)
#     assert np.isclose(svyglm.cov_beta[3, 3], 0.3873394**2)
