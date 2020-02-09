import pytest

import numpy as np
import pandas as pd

from samplics.sae.eb_area_model import AreaModel

milk = pd.read_csv("./tests/sae/milk.csv")

area = milk["SmallArea"]
yhat = milk["yi"]
X = pd.get_dummies(milk["MajorArea"])
sigma2_e = milk["SD"] ** 2


fh_model = AreaModel("FH").fit(area=area, yhat=yhat, X=X, sigma2_e=sigma2_e)