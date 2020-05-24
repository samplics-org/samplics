get_ipython().run_line_magic("load_ext", " lab_black")
import numpy as np
import pandas as pd

import samplics
from samplics.sampling import Sample
from samplics.sae import EbUnitModel


np.random.seed(123)

# model parameters
scale = 1
sigma2e = 0.5 ** 2
sigma2u = 0.15 ** 2

# Population sizes
N = 20000
nb_areas = 80

# Errors generation
error = np.random.normal(loc=0, scale=(scale ** 2) * (sigma2e ** 0.5), size=N)
area = np.sort(np.random.choice(range(1, nb_areas + 1), N))
areas, Nd = np.unique(area, return_counts=True)
random_effects = np.random.normal(loc=0, scale=sigma2u ** (1 / 2), size=nb_areas)
total_error = np.repeat(random_effects, Nd) + error

# Auxiliary information
p1 = 0.3 + 0.5 * np.linspace(1, nb_areas + 1, nb_areas) / nb_areas
p2 = 0.2
X1 = np.array([]).astype(int)
for i, d in enumerate(areas):
    Xk = np.random.binomial(1, p=p1[i], size=Nd[i])
    X1 = np.append(X1, Xk)
X2 = np.random.binomial(1, p=p2, size=N)
X = np.column_stack((np.ones(N), X1, X2))

beta = np.array([3, 0.03, -0.04])
Y = np.matmul(X, beta) + total_error
income = np.exp(Y)

# Create a dataframe for the population data
census_data = pd.DataFrame(data={"area": area, "X1": X1, "X2": X2, "income": income})

nb_obs = 15
print(f"\nFirst {nb_obs} rows of the population data\n")
census_data.head(nb_obs)


sample_size = 50
stratum = census_data["area"]
unit_id = census_data.index

sae_sample = Sample(method="srs", stratification=True, with_replacement=False)
sample, _, _ = sae_sample.select(unit_id, sample_size, stratum)
sample_data = census_data[sample == 1]

nb_obs = 15
print(f"\nFirst {nb_obs} rows of the sample data\n")
sample_data.head(nb_obs)


# Sample data
areas = sample_data["area"]
income = sample_data["income"]
Xs = sample_data[["X1", "X2"]]

# Out of sample data
outofsample_data = census_data[sample get_ipython().getoutput("= 1]")
arear = outofsample_data["area"]
Xr = outofsample_data[["X1", "X2"]]


eb_poverty_reml = EbUnitModel(method="REML", boxcox=0)
eb_poverty_reml.fit(income, Xs, areas, intercept=True)

print(f"\nThe estimated fixed effect using REML is: {eb_poverty_reml.fixed_effects}")
print(
    f"\nThe estimated area random effect standard error using REML is: {eb_poverty_reml.re_std}"
)
print(
    f"\nThe estimated area random effect standard error using REML is: {eb_poverty_reml.error_std}"
)


eb_poverty_ml = EbUnitModel(method="ML", boxcox=0)
eb_poverty_ml.fit(income, Xs, areas, intercept=True)

print(f"\nThe ML estimated fixed effect using ML is: {eb_poverty_ml.fixed_effects}")
print(
    f"\nThe ML estimated area random effect standard error using ML is: {eb_poverty_ml.re_std}"
)
print(
    f"\nThe ML estimated area random effect standard error using ML is: {eb_poverty_ml.error_std}"
)


poverty_incidence(y, poverty_line = 12)

poverty_gap(y, poverty_line = 12)


eb_poverty_reml.predict(Xr, arear, intercept = True)



