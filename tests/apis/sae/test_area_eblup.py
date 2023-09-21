# import numpy as np
import polars as pl

from samplics.apis.sae import fit_eblup
from samplics.types import AuxVars, DirectEst, FitMethod


# Import the datasets
milk = pl.read_csv("./tests/sae/milk.csv")


area = milk["SmallArea"]
yhat = milk["yi"]
X = milk.select("MajorArea").to_dummies(drop_first=True)  # Select() returns a DF
sigma_e = milk["SD"]


# Initialize direct estimates
yhat = DirectEst(
    est=milk["yi"], stderr=milk["SD"], ssize=milk["ni"], domain=milk["SmallArea"]
)


# Initialize AuxVars
auxvars = AuxVars(domain=area, auxdata=X)

fit_reml = fit_eblup(y=yhat, x=auxvars, method=FitMethod.reml)

# fh_model_reml.fit(
#     yhat=yhat,
#     X=X,
#     area=area,
#     intercept=False,
#     error_std=sigma_e,
#     tol=1e-4,
# )
# fh_model_reml.predict(X=X, area=area, intercept=False)

breakpoint()
