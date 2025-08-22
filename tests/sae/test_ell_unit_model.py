import numpy as np
import pandas as pd

from samplics.sae.robust_unit_model import EllUnitModel

incomesample = pd.read_csv("./tests/sae/incomedata.csv")

areas = incomesample["prov"]
ys = incomesample["income"]
Xs = incomesample[
    ["age2", "age3", "age4", "age5", "nat1", "educ1", "educ3", "labor1", "labor2"]
].to_numpy()
# X_s = np.insert(X_s, 0, np.ones(X_s.shape[0]), axis=1)

X_outsample = pd.read_csv("./tests/sae/Xoutsamp.csv")

arear = X_outsample["domain"]
Xr = X_outsample[
    ["age2", "age3", "age4", "age5", "nat1", "educ1", "educ3", "labor1", "labor2"]
].to_numpy()
# X_s = np.insert(X_s, 0, np.ones(X_s.shape[0]), axis=1)

np.random.seed(12345)


def pov_gap(y, pov_line):
    return np.mean((y < pov_line) * (pov_line - y) / pov_line)


"""REML Method"""

ell_bhf_reml = EllUnitModel(
    method="REML",
    boxcox=0,
    constant=3600.5,
)
ell_bhf_reml.fit(ys, Xs, areas, intercept=True)
ell_bhf_reml.predict(Xr, arear, pov_gap, 10, show_progress=False, pov_line=6477.484)


def test_ell_bhf_reml():
    assert ell_bhf_reml.method == "REML"


def test_fixed_effects_ell_bhf_reml():
    assert np.isclose(
        ell_bhf_reml.fixed_effects,
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


def test_re_std_ell_bhf_reml():
    assert np.isclose(ell_bhf_reml.re_std**2, 0.009116, atol=1e-1)


def test_error_var_ell_bhf_reml():
    assert np.isclose(ell_bhf_reml.error_std**2, 0.170677, atol=1e-1)


"""ML Method"""

ell_bhf_ml = EllUnitModel(
    method="ML",
    boxcox=0,
    constant=3600.5,
)
ell_bhf_ml.fit(ys, Xs, areas, intercept=True)
ell_bhf_ml.predict(Xr, arear, pov_gap, 10, show_progress=False, pov_line=6477.484)


def test_ell_bhf_ml():
    assert ell_bhf_ml.method == "ML"


def test_fixed_effects_ell_bhf_ml():
    assert np.isclose(
        ell_bhf_ml.fixed_effects,
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


def test_re_std_ell_bhf_ml():
    assert np.isclose(ell_bhf_ml.re_std**2, 0.009116, atol=1e-1)


def test_error_var_ell_bhf_ml():
    assert np.isclose(ell_bhf_ml.error_std**2, 0.170677, atol=1e-1)


"""MOM Method"""

ell_bhf_mom = EllUnitModel(
    method="MOM",
    boxcox=0,
    constant=3600.5,
)
ell_bhf_mom.fit(ys, Xs, areas, intercept=True)
ell_bhf_mom.predict(Xr, arear, pov_gap, 10, show_progress=False, pov_line=6477.484)


def test_ell_bhf_mom():
    assert ell_bhf_mom.method == "MOM"


def test_fixed_effects_ell_bhf_mom():
    assert np.isclose(
        ell_bhf_mom.fixed_effects,
        np.array(
            [
                9.55711372,
                -0.0285726,
                -0.03091678,
                0.06122963,
                0.02671279,
                -0.04012137,
                -0.15192083,
                0.2792275,
                0.16319449,
                -0.05242666,
            ]
        ),
        atol=1e-4,
    ).all()
