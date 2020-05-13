"""Provides the custom types used throughout the modules.
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd


Array = Union[np.ndarray, pd.Series]
Number = Union[float, int]
StringNumber = Union[str, float, int]
DictStrNum = Dict[StringNumber, Number]
