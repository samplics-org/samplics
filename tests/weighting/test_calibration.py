import numpy as np
import pandas as pd

import samplics
from samplics.weighting import SampleWeight


X = pd.DataFrame(
    {
        "A": [
            "Age1",
            "Age2",
            "Age2",
            "Age1",
            "Age1",
            "Age1",
            "Age2",
            "Age2",
            "Age1",
            "Age2",
            "Age1",
            "Age2",
            "Age1",
            "Age1",
            "Age1",
            "Age1",
            "Age1",
            "Age2",
            "Age1",
            "Age2",
        ],
        "B": [
            "Race1",
            "Race1",
            "Race2",
            "Race1",
            "Race2",
            "Race2",
            "Race1",
            "Race2",
            "Race1",
            "Race2",
            "Race1",
            "Race1",
            "Race2",
            "Race1",
            "Race2",
            "Race2",
            "Race1",
            "Race2",
            "Race1",
            "Race2",
        ],
        "C": [1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2],
    }
)

x_array, x_dict = SampleWeight().calib_covariates(["A", "B"], None, data=X)

x_dict["Age1_&_Race1"] = 10
x_dict["Age1_&_Race2"] = 5
x_dict["Age2_&_Race1"] = 5
x_dict["Age2_&_Race2"] = 10
# x_dict["C"] = 25
print(x_array)
print(x_dict)

wgt = np.ones(x_array.shape[0])
print(wgt.size)

# calib_wgt = SampleWeight().calibrate(wgt, x_array, control=x_dict)

xx = np.transpose(x_array) * wgt[:]
print(xx.shape)
print(np.sum(np.transpose(x_array) * wgt[:], axis=2))

