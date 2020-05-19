import numpy as np
import pandas as pd

import samplics 
from samplics.sae import EblupAreaModel


milk_exp = pd.read_csv("../../../datasets/docs/expenditure_on_milk.csv")

nb_obs = 15
print(f"\nFirst {nb_obs} observations of the Milk Expendure dataset\n")
milk_exp.tail(nb_obs)


area = milk_exp["small_area"]
yhat = milk_exp["direct_est"]
X = pd.get_dummies(milk_exp["major_area"],drop_first=True)
sigma_e = milk_exp["std_error"]

## REML method
fh_model_reml = EblupAreaModel(method="REML")
fh_model_reml.fit(
    yhat=yhat, X=X, area=area, error_std=sigma_e, intercept=True, tol=1e-8,
)

print(f"\nThe estimated fixed effects are: {fh_model_reml.fixed_effects}")
print(f"\nThe estimated standard error of the area random effects is: {fh_model_reml.re_std}")
print(f"\nThe convergence statistics are: {fh_model_reml.convergence}")
print(f"\nThe goodness of fit statistics are: {fh_model_reml.goodness}\n")


fh_model_reml.predict(
    X=X, area=area, intercept=True
)

import pprint
pprint.pprint(fh_model_reml.area_est)


milk_est_reml = fh_model_reml.to_dataframe(col_names = ["small_area", "eblup_estimate", "eblup_mse"])
print(f"\nThe dataframe version of the area level estimates:\n\n {milk_est_reml}")


## ML method
fh_model_ml = EblupAreaModel(method="ML")
fh_model_ml.fit(
    yhat=yhat, X=X, area=area, error_std=sigma_e, intercept=True, tol=1e-8,
)

milk_est_ml = fh_model_ml.predict(
    X=X, area=area, intercept=True
)

milk_est_ml = fh_model_ml.to_dataframe(col_names = ["small_area", "eblup_estimate", "eblup_mse"])


print(f"\nThe dataframe version of the ML area level estimates:\n\n {milk_est_ml}")


## FH method
fh_model_fh = EblupAreaModel(method="FH")
fh_model_fh.fit(
    yhat=yhat, X=X, area=area, error_std=sigma_e, intercept=True, tol=1e-8,
)

milk_est_fh = fh_model_fh.predict(
    X=X, area=area, intercept=True
)

milk_est_fh = fh_model_fh.to_dataframe(col_names = ["small_area", "eblup_estimate", "eblup_mse"])


print(f"\nThe dataframe version of the ML area level estimates:\n\n {milk_est_fh}")



