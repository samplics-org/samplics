import numpy as np
import pandas as pd


def yrbs():
    # Read the the Youth Risk Behaviors survey, 2015 (Only one variable extracted)
    yrbs_data = pd.read_csv("./survmeth/datasets/yrbs.csv")

    # Extraction of the design features
    yrbs_y = yrbs_data["qn8"]
    yrbs_y.replace(2, 0, inplace=True)
    yrbs_stratum = yrbs_data["stratum"]
    yrbs_psu = yrbs_data["psu"]
    yrbs_weight = yrbs_data["weight"]

    return yrbs_data, yrbs_y, yrbs_weight, yrbs_stratum, yrbs_psu
