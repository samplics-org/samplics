import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import EbUnitModel


incomesample = pd.read_csv("./datasets/sae/incomedata.csv")

areas = incomesample["prov"]
ys = incomesample["income"]
Xs = incomesample[
    ["age2", "age3", "age4", "age5", "nat1", "educ1", "educ3", "labor1", "labor2"]
].to_numpy()
# X_s = np.insert(X_s, 0, np.ones(X_s.shape[0]), axis=1)

X_outsample = pd.read_csv("./datasets/sae/Xoutsamp.csv")

arear = X_outsample["domain"]
Xr = X_outsample[
    ["age2", "age3", "age4", "age5", "nat1", "educ1", "educ3", "labor1", "labor2"]
].to_numpy()
# X_s = np.insert(X_s, 0, np.ones(X_s.shape[0]), axis=1)

np.random.seed(12345)


def pov_gap(y, pov_line):
    return np.mean((y < pov_line) * (pov_line - y) / pov_line)


"""REML Method"""

eb_bhf_reml = EbUnitModel(method="REML", boxcox=0, constant=3600.5,)
eb_bhf_reml.fit(ys, Xs, areas, intercept=True)

eb_bhf_reml.predict(Xr, arear, pov_gap, 10, show_progress=False, pov_line=6477.484)

eb_bhf_reml.bootstrap_mse(Xr, arear, pov_gap, 10, show_progress=False, pov_line=6477.484)


def test_eb_bhf_reml():
    assert eb_bhf_reml.method == "REML"


def test_fixed_effects_eb_bhf_reml():
    assert np.isclose(
        eb_bhf_reml.fixed_effects,
        np.array(
            [
                9.537283,
                -0.027813,
                -0.027413,
                0.074673,
                0.043535,
                -0.028042,
                -0.159866,
                0.283830,
                0.163679,
                -0.056200,
            ]
        ),
        atol=1e-4,
    ).all()


def test_re_std_eb_bhf_reml():
    assert np.isclose(eb_bhf_reml.re_std ** 2, 0.009116, atol=1e-1)


def test_error_var_eb_bhf_reml():
    assert np.isclose(eb_bhf_reml.error_std ** 2, 0.170677, atol=1e-1)


"""ML Method"""

eb_bhf_ml = EbUnitModel(method="ML", boxcox=0, constant=3600.5,)
eb_bhf_ml.fit(ys, Xs, areas, intercept=True)

eb_bhf_ml.predict(Xr, arear, pov_gap, 10, show_progress=False, pov_line=6477.484)

# eb_bhf_ml.bootstrap_mse(Xr, arear, pov_gap, 10, show_progress=False, pov_line=6477.484)


def test_eb_bhf_ml():
    assert eb_bhf_ml.method == "ML"


def test_fixed_effects_eb_bhf_ml():
    assert np.isclose(
        eb_bhf_ml.fixed_effects,
        np.array(
            [
                9.537283,
                -0.027813,
                -0.027413,
                0.074673,
                0.043535,
                -0.028042,
                -0.159866,
                0.283830,
                0.163679,
                -0.056200,
            ]
        ),
        atol=1e-4,
    ).all()


def test_re_std_eb_bhf_ml():
    assert np.isclose(eb_bhf_ml.re_std ** 2, 0.009116, atol=1e-1)


def test_error_var_eb_bhf_ml():
    assert np.isclose(eb_bhf_ml.error_std ** 2, 0.170677, atol=1e-1)


lmm_pop = pd.read_csv("./tests/sae/simulated_lmm_population_seed12345.csv")

area = lmm_pop["area"]
Y = lmm_pop["Y"]
X = lmm_pop[["X1", "X2", "X3"]]
sample = lmm_pop["sample"]

# Sample data
area_s = area[sample == 1]
y_s = Y[sample == 1]
X_s = X[sample == 1]


# Out of sample data
area_r = area[sample != 1]
X_r = X[sample != 1]


"""REML Method"""


eb_bhf_reml = EbUnitModel(method="REML", boxcox=0, constant=10,)
eb_bhf_reml.fit(y_s, X_s, area_s, tol=1e-6, intercept=True)


def test_eb_bhf_reml():
    assert eb_bhf_reml.method == "REML"


def test_fixed_effects_eb_bhf_reml():
    assert np.isclose(
        eb_bhf_reml.fixed_effects,
        np.array([5.18988986, 2.89804078, -3.00472657, 2.82705747]),
        atol=1e-4,
    ).all()


def test_re_std_eb_bhf_reml():
    assert np.isclose(eb_bhf_reml.re_std, 0.259776, atol=1e-4)


def test_error_var_eb_bhf_reml():
    assert np.isclose(eb_bhf_reml.error_std, 0.957548, atol=1e-4)


"""ML Method"""


eb_bhf_ml = EbUnitModel(method="ML", boxcox=0, constant=10,)
eb_bhf_ml.fit(y_s, X_s, area_s, tol=1e-6, intercept=True)


def test_eb_bhf_ml():
    assert eb_bhf_ml.method == "ML"


def test_fixed_effects_eb_bhf_ml():
    assert np.isclose(
        eb_bhf_ml.fixed_effects, np.array([5.189840, 2.898447, -3.005277, 2.827395]), atol=1e-4,
    ).all()


def test_re_std_eb_bhf_ml():
    assert np.isclose(eb_bhf_reml.re_std, 0.259776, atol=1e-4)


def test_error_var_eb_bhf_ml():
    assert np.isclose(eb_bhf_reml.error_std, 0.957584, atol=1e-4)
