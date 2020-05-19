import numpy as np
import pandas as pd

import samplics 
from samplics.sae import EblupUnitModel


countycropareas = pd.read_csv("../../../datasets/docs/countycropareas.csv")

nb_obs = 15
print(f"\nFirst {nb_obs} observations from the unit (segment) level crop areas data\n")
countycropareas.head(nb_obs)


countycropareas_means = pd.read_csv("../../../datasets/docs/countycropareas_means.csv")

print(f"\nCounty level crop areas averages\n")
countycropareas_means.head(15)


areas = countycropareas["county_id"]
ys = countycropareas["corn_area"]
Xs = countycropareas[["corn_pixel", "soybeans_pixel"]]
Xs_mean = countycropareas_means[["ave_corn_pixel", "ave_corn_pixel"]]
samp_size = countycropareas_means[["samp_segments"]]
pop_size = countycropareas_means[["pop_segments"]]

"""REML Method"""
eblup_bhf_reml = EblupUnitModel()
eblup_bhf_reml.fit(
    ys, Xs, areas,
)

eblup_bhf_reml.predict(Xs_mean, areas)

corn_est_reml = eblup_bhf_reml.to_dataframe()

print(corn_est_reml)






