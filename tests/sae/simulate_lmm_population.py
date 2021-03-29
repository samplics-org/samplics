import numpy as np
import pandas as pd
import statsmodels.api as sm


np.random.seed(12345)

# model parameters
scale = 1
sigma2e = 1 ** 2
sigma2u = 0.25 ** 2
# print(sigma2u / (sigma2u + sigma2e / 50))

# Population sizes
N = 6000
nb_areas = 20


error = np.random.normal(loc=0, scale=(scale ** 2) * (sigma2e ** 0.5), size=N)
area = np.sort(np.random.choice(range(1, nb_areas + 1), N))
areas, Nd = np.unique(area, return_counts=True)

random_effects = np.random.normal(loc=0, scale=sigma2u ** (1 / 2), size=nb_areas)
total_error = np.repeat(random_effects, Nd) + error

# Auxiliary information
p1 = 0.3 + areas / np.max(areas)
p1 = p1 / sum(p1)
p1 = p1 / (1.2 * max(p1))

X1 = np.array([])
for k, s in enumerate(Nd):
    Xk = np.random.binomial(1, p=p1[k], size=s)
    X1 = np.append(X1, Xk)
X2 = 0.01 + np.random.beta(0.5, 1, N)
X3 = np.random.binomial(1, p=0.6, size=N)

X = np.column_stack((np.ones(N), X1, X2, X3))
beta = np.array([5, 3, -3, 3])
Y = np.exp(np.matmul(X, beta) + total_error)

average_nd = 30
sample = np.random.binomial(1, average_nd * nb_areas / N, N)

# simulated_lmm_population = pd.DataFrame(
#     np.column_stack((Y, area, X[:, 1:4], sample)), columns=["Y", "area", "X1", "X2", "X3", "sample"],
# ).astype({"area": "int16", "X1": "int8", "X3": "int8", "sample": "int8"})

# simulated_lmm_population.to_csv("./tests/sae/simulated_lmm_population_seed12345.csv", index=False)
