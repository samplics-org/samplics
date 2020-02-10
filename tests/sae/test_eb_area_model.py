import pytest

import numpy as np
import pandas as pd

from samplics.sae.eb_area_model import AreaModel

milk = pd.read_csv("./tests/sae/milk.csv")

area = milk["SmallArea"]
yhat = milk["yi"]
X = pd.get_dummies(milk["MajorArea"])
X.loc[:, 1] = 1
# print(X)
sigma2_e = milk["SD"] ** 2


fh_model = AreaModel(model="FH")
fh_model.predict(
    y_s=yhat, X_s=X, X_r=None, area_s=area, area_r=None, sigma2_e=sigma2_e, method="FH"
)

# print(f"Fixed effects coefficients: {fh_model.fe_coef}")
# print(f"Fixed effects covariance: {np.diag(fh_model.fe_cov)**(1/2)}\n")

# print(f"Random effects coefficients: {fh_model.re_coef}")
# print(f"Random effects covariance: {fh_model.re_cov**(1/2)}\n")

# print(f"Convergence: {fh_model.convergence}\n")

# print(f"Point Estimates: {fh_model.point_est}")
# print(f"Mean squared error : {fh_model.mse}\n")

print(fh_model)
