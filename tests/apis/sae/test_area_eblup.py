# import numpy as np
import polars as pl

from samplics.types import AuxVars, DirectEst


# Import the datasets
milk = pl.read_csv("./tests/sae/milk.csv")


area = milk["SmallArea"]
yhat = milk["yi"]
X = milk.select("SmallArea").to_dummies(drop_first=True)  # Select() returns a DF
sigma_e = milk["SD"]


# Initialize direct estimates
yhat = DirectEst(
    area=milk["SmallArea"], est=milk["yi"], stderr=milk["SD"], ssize=milk["ni"]
)


# Initialize AuxVars

auxvars = AuxVars(area=area, auxdata=X)

# Fit the model
# fh_model_reml = EblupAreaModel(method="REML")
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
