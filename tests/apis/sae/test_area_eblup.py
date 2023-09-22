# import numpy as np
import polars as pl

from samplics.apis.sae import fit_eblup
from samplics.types import AuxVars, DirectEst, FitMethod


# Import the datasets
milk = pl.read_csv("./tests/sae/milk.csv")

area = milk["SmallArea"]
yhat = milk["yi"]
sigma_e = milk["SD"]
X = milk.select("MajorArea").to_dummies(drop_first=True)  # Select() returns a DF
n = milk["ni"]

# Initialize direct estimates
yhat = DirectEst(est=yhat, stderr=sigma_e, ssize=n, domain=area)

# Initialize AuxVars
auxvars = AuxVars(domain=area, auxdata=X)

# Fit the linear mixed model
fit_reml = fit_eblup(y=yhat, x=auxvars, method=FitMethod.reml)

# Predict the small area estimates
# fh_model_reml.predict(X=X, area=area, intercept=False)

breakpoint()
