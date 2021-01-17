"""Provides the custom types used throughout the modules.
"""

from typing import Dict, TypeVar, Union

import numpy as np
import pandas as pd


Array = Union[np.ndarray, pd.Series, list, tuple]
Series = Union[pd.Series, list, tuple]

Number = TypeVar("Number", int, float)  # Union[float, int]
StringNumber = TypeVar("StringNumber", str, float, int)  # Union[str, float, int]

DictStrNum = Dict[StringNumber, Number]
