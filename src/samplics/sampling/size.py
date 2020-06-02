"""Sample size calculation module 

"""

from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd

import math

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber


class SampleSize:
    """*SampleSize* implements sample size calculation methods
    """

    def __init__(self, method: str, stratification: str) -> None:
        pass

    def icc(self):
        pass

    def deff(self):
        pass

    def allocate(
        self,
        stratum: Optional[Array] = None,
        Number_strata: Optional[int] = None,
        deff: Union[Dict[Any, float], float] = 1.0,
        response_rate: Union[Dict[Any, float], float] = 1.0,
    ) -> None:
        pass

    def to_dataframe(self) -> pd.DataFrame:
        pass
