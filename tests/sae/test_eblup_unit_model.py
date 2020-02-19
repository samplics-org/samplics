import pytest

import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import UnitModel

from samplics.utils.basic_functions import BoxCox

cornsoybean = pd.read_csv("./tests/sae/cornsoybean.csv")
cornsoybean_mean = pd.read_csv("./tests/sae/cornsoybeanmeans.csv")

# print(cornsoybean.head(20))
# print(cornsoybean_mean.head(200))

area_s = cornsoybean["County"]

y_s = cornsoybean["CornHec"]
X_s = cornsoybean[["CornPix", "SoyBeansPix"]]

X_smean = cornsoybean_mean[["MeanCornPixPerSeg", "MeanSoyBeansPixPerSeg"]]


eblup_bhf = UnitModel()
eblup_bhf.fit(y_s, X_s, area_s)

# print(eblup_bhf.fe_coef)
# print(eblup_bhf.goodness)
# print(eblup_bhf.convergence)

eblup_bhf.predict(X_smean, np.unique(area_s))

results = pd.DataFrame(
    data={
        "area": eblup_bhf.area_s,
        "y_pred": eblup_bhf.y_predicted,
        "random effects": eblup_bhf.random_effect,
    }
)

print(eblup_bhf.fixed_effects)
print(results)
