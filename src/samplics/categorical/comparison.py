"""Comparison module 

The module implements comparisons of groups.

"""

from __future__ import annotations
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from samplics.utils.checks import assert_probabilities
from samplics.utils.formats import numpy_array, remove_nans
from samplics.utils.types import Array, Number, Series, StringNumber

from samplics.estimation import TaylorEstimator


class Ttest:
    def __init__(self, type: str, paired: Optional[bool] = None, alpha: float = 0.05) -> None:

        if type.lower not in ("one-sample", "two-sample", "many-sample"):
            raise ValueError(
                "Parameter 'type' must be equal to 'one-sample', 'two-sample' or 'many-sample'!"
            )
        assert_probabilities(alpha)

        self.type = type.lower()
        self.paired = paired
        self.alpha = alpha

    def _one_sample(y: np.ndarray, group: Array) -> None:
        pass

    def compare(
        self,
        y: Series,
        known_mean: Number = None,
        group: Optional[Array] = None,
        samp_weight: Array = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
        deff: bool = False,
        coef_variation: bool = False,
        remove_nan: bool = False,
    ) -> None:

        if known_mean is None and group is None:
            raise AssertionError("Parameters 'known_mean' or 'group' must be provided!")
        if known_mean is not None and group is not None:
            raise AssertionError("Only one parameter 'known_mean' or 'group' should be provided!")

        y = numpy_array(y)

        if self.type == "one-sample":
            self._one_sample()
        elif self.type == "two-sample":
            pass
        elif self.type == "many-sample":
            pass
        else:
            raise ValueError("type must be equal to 'one-sample', 'two-sample' or 'many-sample'!")

    def by(self, by_var: Array) -> Dict[StringNumber, Ttest]:
        pass