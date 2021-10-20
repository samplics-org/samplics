import numpy as np
import pandas as pd

from samplics.regression import SurveyGLM


api_strat = pd.read_csv("./tests/regression/api_strat.csv")

y = api_strat["api00"]
x = api_strat[["ell", "meals", "mobility"]]
x.insert(0, "intercept", 1)
stratum = api_strat["stype"]
psu = api_strat["dnum"]
weight = api_strat["pw"]


def test_survey_glm_not_stratified_srs():
    svyglm = SurveyGLM()
    svyglm.estimate(y=y, x=x, samp_weight=weight)

    assert np.isclose(svyglm.beta[0], 820.8873)
    assert np.isclose(svyglm.beta[1], -0.4805866)
    assert np.isclose(svyglm.beta[2], -3.141535)
    assert np.isclose(svyglm.beta[3], 0.2257132)

    assert np.isclose(svyglm.cov_beta[0, 0], 10.97091 ** 2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.3971755 ** 2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.2917333 ** 2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.4012498 ** 2)


def test_survey_glm_stratified_srs():
    svyglm = SurveyGLM()
    svyglm.estimate(y=y, x=x, samp_weight=weight, stratum=stratum)

    assert np.isclose(svyglm.beta[0], 820.8873)
    assert np.isclose(svyglm.beta[1], -0.4805866)
    assert np.isclose(svyglm.beta[2], -3.141535)
    assert np.isclose(svyglm.beta[3], 0.2257132)

    assert np.isclose(svyglm.cov_beta[0, 0], 10.25649 ** 2)
    assert np.isclose(svyglm.cov_beta[1, 1], 0.3977075 ** 2)
    assert np.isclose(svyglm.cov_beta[2, 2], 0.2883001 ** 2)
    assert np.isclose(svyglm.cov_beta[3, 3], 0.4026908 ** 2)
