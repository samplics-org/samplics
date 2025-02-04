import numpy as np
import pandas as pd

from samplics.regression import SurveyGLM
from samplics.utils.types import ModelType


api_strat = pd.read_csv("./tests/regression/api_strat.csv")

y = api_strat["api00"]
y_bin = api_strat["api00"] > 743  # api_strat["api00"].quantile(0.75)
x = api_strat[["ell", "meals", "mobility"]]
x.insert(0, "intercept", 1)
stratum = api_strat["stype"]
psu = api_strat["dnum"]
weight = api_strat["pw"]


## Logistic regression


def test_reg_logistic_not_stratified():
    svyglm = SurveyGLM()
    svyglm.estimate(model=ModelType.LOGISTIC, y=y_bin, x=x, samp_weight=weight)

    assert np.isclose(svyglm.beta[0], 2.665756)
    assert np.isclose(svyglm.beta[1], -0.0370325)
    assert np.isclose(svyglm.beta[2], -0.0846879)
    assert np.isclose(svyglm.beta[3], -0.0044286)

    assert np.isclose(svyglm.cov_beta[0, 0], 0.4718306**2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.0336339**2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.0146019**2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.0147885**2)


def test_reg_logistic_psu_not_stratified():
    svyglm = SurveyGLM()
    svyglm.estimate(model=ModelType.LOGISTIC, y=y_bin, x=x, samp_weight=weight, psu=psu)

    assert np.isclose(svyglm.beta[0], 2.665756)
    assert np.isclose(svyglm.beta[1], -0.0370325)
    assert np.isclose(svyglm.beta[2], -0.0846879)
    assert np.isclose(svyglm.beta[3], -0.0044286)

    assert np.isclose(svyglm.cov_beta[0, 0], 0.4949581**2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.0318235**2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.0143854**2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.0153727**2)


def test_reg_logistic_stratified():
    svyglm = SurveyGLM()
    svyglm.estimate(
        model=ModelType.LOGISTIC, y=y_bin, x=x, samp_weight=weight, stratum=stratum
    )

    assert np.isclose(svyglm.beta[0], 2.665756)
    assert np.isclose(svyglm.beta[1], -0.0370325)
    assert np.isclose(svyglm.beta[2], -0.0846879)
    assert np.isclose(svyglm.beta[3], -0.0044286)

    assert np.isclose(svyglm.cov_beta[0, 0], 0.4394433**2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.0336371**2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.0143307**2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.0148366**2)


def test_reg_logistic_psu_stratified():
    svyglm = SurveyGLM()
    svyglm.estimate(
        model=ModelType.LOGISTIC,
        y=y_bin,
        x=x,
        samp_weight=weight,
        psu=psu,
        stratum=stratum,
    )

    assert np.isclose(svyglm.beta[0], 2.665756)
    assert np.isclose(svyglm.beta[1], -0.0370325)
    assert np.isclose(svyglm.beta[2], -0.0846879)
    assert np.isclose(svyglm.beta[3], -0.0044286)

    assert np.isclose(svyglm.cov_beta[0, 0], 0.4723953**2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.0324191**2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.0142838**2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.015463**2)


## Linear regression


def test_reg_linear_not_stratified():
    svyglm = SurveyGLM()
    svyglm.estimate(model=ModelType.LINEAR, y=y, x=x, samp_weight=weight)

    assert np.isclose(svyglm.beta[0], 820.8873)
    assert np.isclose(svyglm.beta[1], -0.4805866)
    assert np.isclose(svyglm.beta[2], -3.141535)
    assert np.isclose(svyglm.beta[3], 0.2257132)

    assert np.isclose(svyglm.cov_beta[0, 0], 10.97091**2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.3971755**2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.2917333**2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.4012498**2)


def test_reg_linear_psu_not_stratified():
    svyglm = SurveyGLM()
    svyglm.estimate(model=ModelType.LINEAR, y=y, x=x, samp_weight=weight, psu=psu)

    assert np.isclose(svyglm.beta[0], 820.8873)
    assert np.isclose(svyglm.beta[1], -0.4805866)
    assert np.isclose(svyglm.beta[2], -3.141535)
    assert np.isclose(svyglm.beta[3], 0.2257132)

    assert np.isclose(svyglm.cov_beta[0, 0], 11.65382**2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.4292165**2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.285711**2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.4159033**2)


def test_reg_linear_stratified():
    svyglm = SurveyGLM()
    svyglm.estimate(
        model=ModelType.LINEAR, y=y, x=x, samp_weight=weight, stratum=stratum
    )

    assert np.isclose(svyglm.beta[0], 820.8873)
    assert np.isclose(svyglm.beta[1], -0.4805866)
    assert np.isclose(svyglm.beta[2], -3.141535)
    assert np.isclose(svyglm.beta[3], 0.2257132)

    assert np.isclose(svyglm.cov_beta[0, 0], 10.25649**2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.3977075**2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.2883001**2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.4026908**2)


def test_reg_linear_psu_stratified():
    svyglm = SurveyGLM()
    svyglm.estimate(
        model=ModelType.LINEAR, y=y, x=x, samp_weight=weight, psu=psu, stratum=stratum
    )

    assert np.isclose(svyglm.beta[0], 820.8873)
    assert np.isclose(svyglm.beta[1], -0.4805866)
    assert np.isclose(svyglm.beta[2], -3.141535)
    assert np.isclose(svyglm.beta[3], 0.2257132)

    assert np.isclose(svyglm.cov_beta[0, 0], 10.10296**2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.4109377**2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.2743919**2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.3873394**2)
