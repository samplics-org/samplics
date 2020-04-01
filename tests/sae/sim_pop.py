import numpy as np
import statsmodels.api as sm

from samplics.sae.eb_unit_model import UnitModel

np.random.seed(531451)

# model parameters
scale = 1
sigma2e = 1 ** 2
sigma2u = 0.25 ** 2
# print(sigma2u / (sigma2u + sigma2e / 50))

# Population sizes
N = 10_000_000
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
    # print(f"Proportion for Area {k} is {np.mean(Xk)}")

X2 = 0.01 + np.random.beta(0.5, 1, N)
# print(f"{np.min(X2)}, {np.mean(X2)}, {np.median(X2)}, {np.max(X2)}")
X3 = np.random.binomial(1, p=0.6, size=N)

X = np.column_stack((np.ones(N), X1, X2, X3))
# print(X)

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
# print(areas.shape)
# print(nd)

# basic_model = sm.MixedLM(y_s, X_s, area_s)
# basic_fit = basic_model.fit(reml=True, full_output=True)

# print(f"Fixed effects: {basic_fit.fe_params}")

# print(f"sigma_e: {basic_fit.scale}")
# print(f"sigma_u: {float(basic_fit.cov_re)**0.5}")


eblup_bhf_reml = UnitModel()
eblup_bhf_reml.fit(y_s, X_s, area_s, intercept=False)

print(f"Fixed effects: {eblup_bhf_reml.fixed_effects}")
print(f"Sigma u: {eblup_bhf_reml.re_std}")
print(f"Sigma e: {eblup_bhf_reml.error_var**0.5}")
