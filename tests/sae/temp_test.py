import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import UnitModel

from samplics.utils.basic_functions import BoxCox

cornsoybean = pd.read_csv("./tests/sae/cornsoybean.csv")
cornsoybean_mean = pd.read_csv("./tests/sae/cornsoybeanmeans.csv")

cornsoybean = cornsoybean.sample(frac=1)

area_s = cornsoybean["County"]

y_s = cornsoybean["CornHec"]
X_s = cornsoybean[["CornPix", "SoyBeansPix"]]

X_smean = cornsoybean_mean[["MeanCornPixPerSeg", "MeanSoyBeansPixPerSeg"]]

samp_size = np.array([1, 1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 6])
pop_size = np.array([545, 566, 394, 424, 564, 570, 402, 567, 687, 569, 965, 556])

eblup_bhf_reml = UnitModel()
eblup_bhf_reml.fit(y_s, X_s, area_s)

eblup_bhf_reml.predict(X_s, X_smean, area_s)

print(eblup_bhf_reml.gamma, "\n")

