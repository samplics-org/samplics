import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import EbUnitModel

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


# Molina-Rao Method
def pov_gap(y, pov_line=500):
    return np.mean((y < pov_line) * (pov_line - y) / pov_line)


# eb_bhf_reml = EbUnitModel(method="REML", boxcox=0, constant=10, indicator=pov_gap)
# eb_bhf_reml.fit(y_s, X_s, area_s, intercept=True)

# print(f"EB fixed effects: {eb_bhf_reml.fixed_effects}")
# print(f"EB sigma u: {eb_bhf_reml.re_std}")
# print(f"EB sigma e: {eb_bhf_reml.error_std}\n")

# eb_bhf_reml.predict(200, pov_gap, X_r, area_r)
# print(f"EB area parameter:\n {eb_bhf_reml.area_est}\n")

# mse = eb_bhf_reml.bootstrap_mse(50, pov_gap, X_r, area_r,)
# print(f"EB area mse:\n {mse}\n")


"""REML Method"""


eb_bhf_reml = EbUnitModel(method="REML", boxcox=0, constant=10, )
eb_bhf_reml.fit(y_s, X_s, area_s, tol=1e-6, intercept=True)


def test_eb_bhf_reml():
    assert eb_bhf_reml.method == "REML"


def test_fixed_effects_eb_bhf_reml():
    assert np.isclose(
        eb_bhf_reml.fixed_effects,
        np.array([5.18988986, 2.89804078, -3.00472657, 2.82705747]),
        atol=1e-1,
    ).all()


def test_re_std_eb_bhf_reml():
    assert np.isclose(eb_bhf_reml.re_std, 0.2500, atol=1e-1)


def test_error_var_eb_bhf_reml():
    assert np.isclose(eb_bhf_reml.error_std, 1.0000, atol=1e-1)
