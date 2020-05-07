import pytest

import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import EblupUnitLevel


cornsoybean = pd.read_csv("./tests/sae/cornsoybean.csv")
cornsoybean_mean = pd.read_csv("./tests/sae/cornsoybeanmeans.csv")

cornsoybean = cornsoybean.sample(frac=1)

area_s = cornsoybean["County"]

y_s = cornsoybean["CornHec"]
X_s = cornsoybean[["CornPix", "SoyBeansPix"]]

X_smean = cornsoybean_mean[["MeanCornPixPerSeg", "MeanSoyBeansPixPerSeg"]]

samp_size = np.array([1, 1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 6])
pop_size = np.array([545, 566, 394, 424, 564, 570, 402, 567, 687, 569, 965, 556])

"""REML Method"""
eblup_bhf_reml = EblupUnitLevel()
eblup_bhf_reml.fit(y_s, X_s, area_s, tol=1e-9)

eblup_bhf_reml.predict(X_smean, area_s)


def test_eblup_bhf_reml():
    assert eblup_bhf_reml.method == "REML"


def test_fixed_effects_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.fixed_effects, np.array([17.96398, 0.3663352, -0.0303638]), atol=1e-6
    ).all()


def test_fe_std_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.fe_std, np.array([30.986801, 0.065101, 0.067583]), atol=1e-6
    ).all()


# def test_gamma_bhf_reml():
#    assert np.isclose(
#        eblup_bhf_reml.gamma, np.array([]), atol=1e-6
#    ).all()


def test_random_effects_bhf_reml():
    assert np.isclose(
        eblup_bhf_reml.random_effects,
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
        atol=1e-6,
    ).all()


def test_re_std_bhf_reml():
    assert np.isclose(eblup_bhf_reml.re_std ** 2, 63.3149, atol=1e-6)


def test_error_var_bhf_reml():
    assert np.isclose(eblup_bhf_reml.error_std ** 2, 297.7128, atol=1e-6)


def test_goodness_of_fit_bhf_reml():
    assert np.isclose(eblup_bhf_reml.goodness["loglike"], -161.005759)
    assert np.isclose(eblup_bhf_reml.goodness["AIC"], 326.011518)
    assert np.isclose(eblup_bhf_reml.goodness["BIC"], 329.064239)


def test_convergence_bhf_reml():
    assert eblup_bhf_reml.convergence["achieved"] == True
    assert eblup_bhf_reml.convergence["iterations"] == 4


def test_area_estimate_bhf_reml():
    assert np.isclose(
        np.array(list(eblup_bhf_reml.area_est.values())),
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
        atol=1e-6,
    ).all()


def test_area_mse_bhf_reml():
    assert np.isclose(
        np.array(list(eblup_bhf_reml.area_mse.values())),
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
        atol=1e-6,
    ).all()


eblup_bhf_reml_fpc = EblupUnitLevel()
eblup_bhf_reml_fpc.fit(y_s, X_s, area_s)

eblup_bhf_reml_fpc.predict(X_smean, area_s, pop_size)


def test_y_predicted_bhf_reml_fpc():
    assert np.isclose(
        np.array(list(eblup_bhf_reml_fpc.area_est.values())),
        np.array(
            [
                122.582519,
                123.527414,
                113.034260,
                114.990082,
                137.266001,
                108.980696,
                116.483886,
                122.771075,
                111.564754,
                124.156518,
                112.462566,
                131.251525,
            ]
        ),
        atol=1e-6,
    ).all()


"""ML Method"""
eblup_bhf_ml = EblupUnitLevel(method="ml")
eblup_bhf_ml.fit(
    y_s, X_s, area_s,
)

eblup_bhf_ml.predict(X_smean, area_s)


def test_eblup_bhf_ml():
    assert eblup_bhf_ml.method == "ML"


def test_fixed_effects_bhf_ml():
    assert np.isclose(
        eblup_bhf_ml.fixed_effects, np.array([18.08888, 0.36566, -0.03017]), atol=1e-5
    ).all()


def test_fe_std_bhf_ml():
    assert np.isclose(eblup_bhf_ml.fe_std, np.array([29.81960, 0.06249, 0.06505]), atol=1e-5).all()


def test_random_effects_bhf_ml():
    assert np.isclose(
        eblup_bhf_ml.random_effects,
        np.array(
            [
                1.8322323,
                1.2218437,
                -3.9308431,
                -2.3261989,
                7.2988558,
                3.7065346,
                -2.3371090,
                1.0315879,
                4.4367420,
                -2.5647926,
                -7.7046350,
                -0.6642178,
            ]
        ),
        atol=1e-6,
    ).all()


def test_re_std_bhf_ml():
    assert np.isclose(eblup_bhf_ml.re_std ** 2, 47.79559, atol=1e-4)


def test_error_var_bhf_ml():
    assert np.isclose(eblup_bhf_ml.error_std ** 2, 280.2311, atol=1e-4)


def test_goodness_of_fit_bhf_ml():
    assert np.isclose(eblup_bhf_ml.goodness["loglike"], -159.1981)
    assert np.isclose(eblup_bhf_ml.goodness["AIC"], 328.4, atol=0.1)
    assert np.isclose(eblup_bhf_ml.goodness["BIC"], 336.5, atol=0.1)


def test_convergence_bhf_ml():
    assert eblup_bhf_ml.convergence["achieved"] == True
    assert eblup_bhf_ml.convergence["iterations"] == 3


def test_area_estimate_bhf_ml():
    assert np.isclose(
        np.array(list(eblup_bhf_ml.area_est.values())),
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
        atol=1e-6,
    ).all()


def test_area_mse_bhf_ml():
    assert np.isclose(
        np.array(list(eblup_bhf_reml.area_mse.values())),
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
        atol=1e-6,
    ).all()


eblup_bhf_ml_fpc = EblupUnitLevel(method="ML")
eblup_bhf_ml_fpc.fit(y_s, X_s, area_s)

eblup_bhf_ml_fpc.predict(X_smean, area_s, pop_size)


def test_y_predicted_bhf_ml_fpc():
    assert np.isclose(
        np.array(list(eblup_bhf_ml_fpc.area_est.values())),
        np.array(
            [
                122.582519,
                123.527414,
                113.034260,
                114.990082,
                137.266001,
                108.980696,
                116.483886,
                122.771075,
                111.564754,
                124.156518,
                112.462566,
                131.251525,
            ]
        ),
        atol=1e-6,
    ).all()
