import numpy as np
import pandas as pd

from samplics.sae.eb_unit_model import UnitModelBHF

from samplics.utils.basic_functions import BoxCox

income = pd.read_csv("./tests/sae/income.csv")


y = income["income"]

# y = np.random.normal(size=100000)

print(BoxCox().get_skewness(y))
print(BoxCox().get_kurtosis(y))

print(BoxCox().transform(y, 0))

# BoxCox().plot_skewness(y)

BoxCox().plot_kurtosis(y, -0.1, 2)
