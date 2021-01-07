"""Comparison module 

The module implements comparisons of groups.

"""

from __future__ import annotations
from typing import Dict, List, Optional, Union

import math
import numpy as np
import pandas as pd

from scipy.stats import t

from samplics.utils.checks import assert_probabilities
from samplics.utils.formats import numpy_array, remove_nans
from samplics.utils.types import Array, Number, Series, StringNumber

from samplics.estimation import TaylorEstimator


class Ttest:
    def __init__(self, type: str, paired: Optional[bool] = None, alpha: float = 0.05) -> None:

        if type.lower() not in ("one-sample", "two-sample"):
            raise ValueError("Parameter 'type' must be equal to 'one-sample', 'two-sample'!")
        assert_probabilities(alpha)

        self.type = type.lower()
        self.paired = paired

        self.point_est: Dict[str, Dict[StringNumber, Number]] = {}
        self.stats: Dict[str, Dict[str, Number]] = {}
        self.stderror: Dict[str, Dict[str, Number]] = {}

        self.lower_ci: Dict[str, Dict[str, Number]] = {}
        self.upper_ci: Dict[str, Dict[str, Number]] = {}
        self.deff: Dict[str, Dict[str, Number]] = {}
        self.alpha: float = alpha
        self.design_info: Dict[str, Number] = {}
        self.group_names: List[str] = []
        self.group_levels: Dict[str, StringNumber] = {}

    def _one_sample(
        self,
        y: np.ndarray,
        known_mean: Number = None,
        group: Array = None,
        samp_weight: Array = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
    ) -> None:

        one_sample = TaylorEstimator(parameter="mean", alpha=self.alpha)

        if known_mean is not None:
            one_sample.estimate(
                y=y,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )

            samp_mean = one_sample.point_est["__none__"]
            samp_std_dev = math.sqrt(y.shape[0]) * one_sample.stderror["__none__"]
            samp_t_value = math.sqrt(y.shape[0]) * (samp_mean - known_mean) / samp_std_dev
            left_p_value = t.cdf(samp_t_value, y.shape[0] - 1)

            self.design_info = {
                "number_strata": one_sample.number_strata,
                "number_psus": one_sample.number_psus,
                "number_obs": y.shape[0],
                "design_effect": 0,
                "degrees_of_freedom": one_sample.number_psus - one_sample.number_strata,
            }

            self.stats = {
                "__none__": {
                    "t_value": samp_t_value,
                    "t_df": y.shape[0] - 1,
                    "known_mean": known_mean,
                    "p-value": {
                        "less_than": left_p_value,
                        "greather_than": 1 - left_p_value,
                        "not_equal": 2 * t.cdf(-abs(samp_t_value), y.shape[0] - 1),
                    },
                }
            }

            self.point_est = one_sample.point_est
            self.stderror = one_sample.stderror
            self.lower_ci = one_sample.lower_ci
            self.upper_ci = one_sample.upper_ci

        if group is not None:
            one_sample.estimate(
                y=y,
                domain=group,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )

        breakpoint()

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
        remove_nan: bool = False,
    ) -> None:

        if known_mean is None and group is None:
            raise AssertionError("Parameters 'known_mean' or 'group' must be provided!")
        if known_mean is not None and group is not None:
            raise AssertionError("Only one parameter 'known_mean' or 'group' should be provided!")

        y = numpy_array(y)

        if self.type == "one-sample":
            self._one_sample(
                y=y,
                known_mean=known_mean,
                group=group,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )
        elif self.type == "two-sample":
            pass
        elif self.type == "many-sample":
            pass
        else:
            raise ValueError("type must be equal to 'one-sample', 'two-sample'!")

    def by(self, by_var: Array) -> Dict[StringNumber, Ttest]:
        pass