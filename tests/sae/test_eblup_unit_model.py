import pytest

import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import UnitModel

from samplics.utils.basic_functions import BoxCox

cornsoybean = pd.read_csv("./tests/sae/cornsoybean.csv")
cornsoybean_mean = pd.read_csv("./tests/sae/cornsoybeanmeans.csv")

area_s = cornsoybean["County"]

y_s = cornsoybean["CornHec"]
X_s = cornsoybean[["CornPix", "SoyBeansPix"]]

X_smean = cornsoybean_mean[["MeanCornPixPerSeg", "MeanSoyBeansPixPerSeg"]]


eblup_bhf_reml = UnitModel()
eblup_bhf_reml.fit(y_s, X_s, area_s)

eblup_bhf_reml.predict(X_smean, np.unique(area_s))

results = pd.DataFrame(
    data={
        "area": eblup_bhf_reml.area_s,
        "y_pred": eblup_bhf_reml.y_predicted,
        "random effects": eblup_bhf_reml.random_effect,
    }
)
print(eblup_bhf_reml.mse)


def test_eblup_bhf_reml():
    assert eblup_bhf_reml.model == "BHF"
    assert eblup_bhf_reml.method == "REML"
    assert eblup_bhf_reml.parameter == "mean"


def test_fixed_effects_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.fixed_effects, np.array([17.96398, 0.3663352, -0.0303638]),
    ).all()


def test_fe_std_bhf_reml():
    assert np.isclose(eblup_bhf_reml.fe_std, np.array([30.986801, 0.065101, 0.067583]),).all()


def test_random_effects_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.random_effect,
        np.array(
            [
                2.184574,
                1.475118,
                -4.730863,
                -2.764825,
                8.370915,
                4.274827,
                -2.705540,
                1.156682,
                5.026852,
                -2.883398,
                -8.652532,
                -0.751808,
            ]
        ),
    ).all()


def test_re_std_bhf_reml():
    assert np.isclose(eblup_bhf_reml.re_std, 63.3149)


def test_error_var_bhf_reml():
    assert np.isclose(eblup_bhf_reml.error_var, 297.7128)


def test_goodness_of_fit_bhf_reml():
    assert np.isclose(eblup_bhf_reml.goodness["loglike"], -161.005759)
    assert np.isclose(eblup_bhf_reml.goodness["AIC"], 326.011518)
    assert np.isclose(eblup_bhf_reml.goodness["BIC"], 329.064239)


def test_convergence_bhf_reml():
    assert eblup_bhf_reml.convergence["achieved"] == True
    assert eblup_bhf_reml.convergence["iterations"] == 5


def test_y_predicted_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.y_predicted,
        np.array(
            [
                122.563671,
                123.515159,
                113.090719,
                115.020744,
                137.196212,
                108.945432,
                116.515532,
                122.761482,
                111.530348,
                124.180346,
                112.504727,
                131.257883,
            ]
        ),
    ).all()


def test_mse_bhf_reml():
    assert False
