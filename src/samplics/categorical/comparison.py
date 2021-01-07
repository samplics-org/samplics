"""Comparison module 

The module implements comparisons of groups.

"""

from __future__ import annotations
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from samplics.utils.checks import assert_probabilities
from samplics.utils.formats import numpy_array, remove_nans
from samplics.utils.types import Array, Number, StringNumber

from samplics.estimation import TaylorEstimator


class Ttest:
    def __init__(self, type: str, paired: Optional[bool] = None, alpha: float = 0.05) -> None:

        if type.lower not in ("one-sample", "two-sample", "many-sample"):
            raise ValueError("type must be equal to 'one-sample', 'two-sample' or 'many-sample'!")
        assert_probabilities(alpha)

        self.type = type.lower()
        self.paired = paired
        self.alpha = alpha

    def compare(
        self,
        y: Array,
        group: Array,
        samp_weight: Array = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
        deff: bool = False,
        coef_variation: bool = False,
        remove_nan: bool = False,
    ) -> None:
        pass

    def by(self, by_var: Array) -> Dict[StringNumber, Ttest]:
        pass