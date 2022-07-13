# import pytest


from typing import Optional, Union, Dict

import numpy as np
import pandas as pd


from pydantic.dataclasses import dataclass

from samplics.sampling import SampleSize

# NOT-STRATIFIED Wald's method
# size_mean_nat = SampleSize(parameter="mean", pop_size=10000)
size_mean_nat = SampleSize(param="total")

assert size_mean_nat.param == "total"
# assert size_mean_nat.param
# size_mean_nat.calculate(half_ci=1, target=1, sigma=2)

# print(size_mean_nat.__pydantic_model__.schema())

# breakpoint()
