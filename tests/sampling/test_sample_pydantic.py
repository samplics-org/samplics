# # import pytest


# from typing import Optional, Union, Dict

# import numpy as np
# import pandas as pd


# from samplics.utils import PopParam
# from samplics.sampling import SampleSize


# half_ci2 = {"stratum1": 0.5, "stratum2": 0.5, "stratum3": 0.5}
# sigma2 = {"stratum1": 2, "stratum2": 2, "stratum3": 2}
# # pop_size2 = {"stratum1": 1000, "stratum2": 10000, "stratum3": 10000000}
# resp_rate2 = {"stratum1": 0.4, "stratum2": 1, "stratum3": 0.65}


# # size_str_mean_wald_fpc1 = SampleSize(param=PopParam.mean, strat=True)
# # size_str_mean_wald_fpc1.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=pop_size2)


# # size_str_mean_wald_fpc2 = SampleSize(param=PopParam.mean, strat=True)
# # size_str_mean_wald_fpc2.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=1000)


# # NOT-STRATIFIED Wald's method
# # size_mean_nat = SampleSize(param=PopParam.mean, pop_size=10000)
# size_mean_nat = SampleSize(param=PopParam.total)

# # assert size_mean_nat.param == "total"
# # assert size_mean_nat.param
# size_mean_nat.calculate(half_ci=1, target=1, sigma=2)

# # print(size_mean_nat.__pydantic_model__.schema())

# # breakpoint()
