import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator

from samplics.utils.types import SinglePSUEst

sample_data = pd.DataFrame.from_dict(
    data={
        "region": [1, 1, 1, 2, 2, 3, 3, 4],
        "district": [1, 2, 1, 1, 1, 1, 1, 1],
        "area": [1, 1, 2, 1, 2, 1, 1, 1],
        "wgt": [1.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5],
        "age": [12, 34, 24, 12, 33, 46, 78, 98],
    }
)

svy_total_single_psu = TaylorEstimator(parameter="total")


def test_single_psu_total():
    svy_total_single_psu.estimate(
        y=sample_data["age"],
        samp_weight=sample_data["wgt"],
        stratum=sample_data["region"],
        psu=sample_data["district"],
        single_psu=SinglePSUEst.skip,
        remove_nan=True,
    )
    # breakpoint()


# breakpoint()
