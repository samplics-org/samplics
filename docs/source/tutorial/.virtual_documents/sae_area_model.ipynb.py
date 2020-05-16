import numpy as np
import pandas as pd

import samplics 
from samplics.sae import EblupAreaModel


milk = pd.read_csv("../../../datasets/docs/milk.csv")

milk.head(12)


area = milk["small_area"]
yhat = milk["direct_est"]
X = pd.get_dummies(milk["major_area"])

X.loc[:, 1] = 1
print(X)
sigma_e = milk["std_error"]

## REML method
fh_model_reml = EblupAreaModel(method="REML")
fh_model_reml.fit(
    yhat=yhat, X=X, area=area, error_std=sigma_e, abstol=1e-4,
)
fh_model_reml.predict(
    X=X, area=area,
)





