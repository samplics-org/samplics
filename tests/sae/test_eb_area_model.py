import pytest

import numpy as np
import pandas as pd

from samplics.sae.eb_area_model import AreaModel

milk = pd.read_csv("./tests/sae/milk.csv")

area = milk["SmallArea"]
yhat = milk["yi"]
X = pd.get_dummies(milk["MajorArea"])
X.loc[:, 1] = 1
print(X)
sigma2_e = milk["SD"] ** 2


fh_model = AreaModel("FH")
fh_model.fit(area=area, yhat=yhat, X=X, sigma2_e=sigma2_e)

print(f"Fixed effects coefficients: {fh_model.fe_coef}\n")
print(f"Fixed effects covariance: {np.diag(fh_model.fe_cov)**(1/2)}")
