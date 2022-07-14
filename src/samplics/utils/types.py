"""Provides the custom types used throughout the modules.
"""

from typing import Dict, Union

from enum import Enum

import numpy as np
import pandas as pd


Array = Union[np.ndarray, pd.Series, list, tuple]
Series = Union[pd.Series, list, tuple]

Number = Union[float, int]
StringNumber = Union[str, float, int]

DictStrNum = Dict[StringNumber, Number]
DictStrInt = Dict[StringNumber, int]
DictStrFloat = Dict[StringNumber, float]
DictStrBool = Dict[StringNumber, bool]

# Population parameters
class PopParam(Enum):
    mean = "mean"
    total = "total"
    prop = "proportion"


# Methods for sample size
class SizeMethod(Enum):
    wald = "wald"
    fleiss = "fleiss"
