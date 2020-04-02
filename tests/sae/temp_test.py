import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import UnitModel

sample = pd.read_csv("./tests/sae/UnitLevel_sample_seed531451.csv")

area_s = sample["area"]
y_s = sample["y"]
X_s = sample[["X1", "X2", "X3"]]

eblup_bhf_reml = UnitModel()
eblup_bhf_reml.fit(y_s, X_s, area_s, intercept=True)

print(f"Fixed effects: {eblup_bhf_reml.fixed_effects}")
print(f"Sigma u: {eblup_bhf_reml.re_std}")
print(f"Sigma e: {eblup_bhf_reml.error_std}\n")

population = pd.read_csv("./tests/sae/UnitLevel_pop_seed531451.csv")
#print(population)

Xmean_s = population[["X1", "X2", "X3"]]

eblup_bhf_reml.predict(X_s, Xmean_s, area_s)

print(f"Gamma parameter:\n {eblup_bhf_reml.gamma}\n")

#print(eblup_bhf_reml.y_predicted)

print()
