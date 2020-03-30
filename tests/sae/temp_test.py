import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import UnitModel

from samplics.utils.basic_functions import BoxCox

cornsoybean = pd.read_csv("./tests/sae/cornsoybean.csv")
cornsoybean_mean = pd.read_csv("./tests/sae/cornsoybeanmeans.csv")

cornsoybean = cornsoybean.sample(frac=1)

area_s = cornsoybean["County"].values

y_s = cornsoybean["CornHec"].values
X_s = cornsoybean[["CornPix", "SoyBeansPix"]]

X_smean = cornsoybean_mean[["MeanCornPixPerSeg", "MeanSoyBeansPixPerSeg"]]

samp_size = np.array([1, 1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 6])
pop_size = np.array([545, 566, 394, 424, 564, 570, 402, 567, 687, 569, 965, 556])

# eblup_bhf_reml = UnitModel()
# eblup_bhf_reml.fit(y_s, X_s, area_s)

# eblup_bhf_reml.predict(X_s, X_smean, area_s)

# print(eblup_bhf_reml.gamma, "\n")


def pd_sumby(area_s, y_s):
    df = pd.DataFrame(data={"area": area_s, "y": y_s})
    df_sum = df.groupby("area").sum()
    # print(df_sum)return df_sum
    return df_sum


def np_sumby(area_s, y_s):

    areas = np.unique(area_s)
    sums = np.zeros(areas.size)
    for k, area in enumerate(areas):
        sums[k] = np.sum(y_s[area_s == area])
    
    return sums

print(pd_sumby(area_s, y_s), "\n")
print(np_sumby(area_s, y_s))