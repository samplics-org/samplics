# import numpy as np
import sys

import polars as pl
import pytest

from samplics.apis import fit
from samplics.apis.sae.area_eblup import _predict_eblup

# from samplics.apis.sae import _log_likelihood, fit_eblup, predict_eblup
from samplics.types import AuxVars, DirectEst, FitMethod


# Import the datasets
# iris = pl.read_csv("./tests/apis/sae/iris.csv")
# y = iris["Sepal.Length"]
# x_dummies = iris.select("Species").to_dummies(drop_first=True)
# x = iris.select(["Petal.Width"]).hstack(x_dummies)
# area = np.linspace(1, y.shape[0], num=y.shape[0]).astype(int)
# ssize = (np.random.rand(y.shape[0]) * 30).astype(int)
# stderr = (np.random.rand(y.shape[0]) * y) / 15
# yhat = DirectEst(est=iris["Sepal.Length"], stderr=stderr, ssize=ssize, domain=area)
# auxvars = AuxVars(auxdata=x, domain=area)

# fit_ml = _log_likelihood(
#     method=FitMethod.ml,
#     y=yhat,
#     x=auxvars,
#     sig2_e=
# )
# breakpoint()
# fit_ml = fit_eblup._l(y=yhat, x=auxvars, method=FitMethod.ml)
# fit_reml = fit_eblup(y=yhat, x=auxvars, method=FitMethod.reml)
# breakpoint()


milk = pl.read_csv("./tests/apis/sae/milk.csv")

area = milk["SmallArea"]
yhat = milk["yi"]
sigma_e = milk["SD"]
x = milk.select("MajorArea").to_dummies(drop_first=True)  # Select() returns a DF
n = milk["ni"]

# Initialize direct estimates
yhat = DirectEst(est=yhat, stderr=sigma_e, ssize=n, domain=area)


# Initialize AuxVars
auxvars = AuxVars(x=x, domain=area)

# Fit the linear mixed model
fit_ml = fit(y=yhat, x=auxvars, method=FitMethod.ml)
fit_reml = fit(y=yhat, x=auxvars, method=FitMethod.reml)
# fit_fh = fit_eblup(y=yhat, x=auxvars, method=FitMethod.fh)
# breakpoint()

# Predict the small area estimates
est_milk_reml = _predict_eblup(x=auxvars, fit_eblup=fit_reml, y=yhat)

# est_milk_reml.fit_stats.log_llike

# breakpoint()


@pytest.mark.skipif(
    sys.platform == "linux", reason="Skip dev version on Github (Linux)"
)
def test_sae1():
    assert 1 == 2
