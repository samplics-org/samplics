import numpy as np
import pandas as pd

import statsmodels.api as sm

from samplics.sae.eb_unit_model import UnitModel

np.random.seed(531451)

# model parameters
scale = 1
sigma2e = 1 ** 2
sigma2u = 0.25 ** 2
# print(sigma2u / (sigma2u + sigma2e / 50))

# Population sizes
N = 1_0000_000
nb_areas = 250


error = np.random.normal(loc=0, scale=(scale ** 2) * (sigma2e ** 0.5), size=N)
area = np.sort(np.random.choice(range(1, nb_areas + 1), N))
areas, Nd = np.unique(area, return_counts=True)
# print(areas)

random_effects = np.random.normal(loc=0, scale=sigma2u ** (1 / 2), size=nb_areas)
total_error = np.repeat(random_effects, Nd) + error
# print(total_error)

# Auxiliary information
p1 = 0.3 + areas / np.max(areas)
p1 = p1 / sum(p1)
p1 = p1 / (1.2 * max(p1))

X1 = np.array([])
for k, s in enumerate(Nd):
    Xk = np.random.binomial(1, p=p1[k], size=s)
    X1 = np.append(X1, Xk)
    print(f"Proportion for Area {k} is {np.mean(Xk)}")

X2 = 0.01 + np.random.beta(0.5, 1, N)
# print(f"{np.min(X2)}, {np.mean(X2)}, {np.median(X2)}, {np.max(X2)}")
X3 = np.random.binomial(1, p=0.6, size=N)

X = np.column_stack((np.ones(N), X1, X2, X3))
# print(X)

Xmean = np.zeros((nb_areas, X.shape[1])) * np.nan
for k, d in enumerate(areas):  # can do this faster using pd.groupby().mean()
    print(f"Computing {k}th mean for area {d}")
    Xmean[k, :] = np.mean(X[area == d], axis=0)
# print(Xmean)

beta = np.array([1, 3, -3, 3])

y = np.matmul(X, beta) + total_error
# print(y)

average_nd = 50
sample = np.random.binomial(1, average_nd * nb_areas / N, N)

y_s = y[sample == 1]
X_s = X[sample == 1]
# print(X_s.shape)

area_s = area[sample == 1]
areas, nd = np.unique(area_s, return_counts=True)
print(areas.shape)
print(nd)

basic_model = sm.MixedLM(y_s, X_s, area_s)
basic_fit = basic_model.fit(reml=True, full_output=True)

print(basic_fit.random_effects)

# print(f"Fixed effects: {basic_fit.fe_params}")

# print(f"sigma_e: {basic_fit.scale}")
# print(f"sigma_u: {float(basic_fit.cov_re)**0.5}")

# print(basic_fit.cov_re / (basic_fit.cov_re + basic_fit.scale ** 2 / nd))


# sample_data = pd.DataFrame(
#     np.column_stack((y_s, area_s, X_s[:, 1:4])), columns=["y", "area", "X1", "X2", "X3"],
# ).astype({"area": "int16", "X1": "int8", "X3": "int8"})
# # print(sample_data)

# sample_data.to_csv("./tests/sae/UnitLevel_sample_seed531451.csv", index=False)

# population_data = pd.DataFrame(
#     np.column_stack((areas, Xmean[:, 1:4], Nd)), columns=["area", "X1", "X2", "X3", "Nsize"]
# ).astype({"area": "int16", "Nsize": "int32"})
# # print(population_data)

# population_data.to_csv("./tests/sae/UnitLevel_pop_seed531451.csv", index=False)
