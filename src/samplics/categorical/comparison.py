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
        self.stderror: Dict[str, Dict[str, Number]] = {}
        self.lower_ci: Dict[str, Dict[str, Number]] = {}
        self.upper_ci: Dict[str, Dict[str, Number]] = {}
        self.deff: Dict[str, Dict[str, Number]] = {}
        self.alpha: float = alpha
        # self.design_info: Dict[str, Number] = {}
        self.group_names: List[str] = []
        self.group_levels: Dict[str, StringNumber] = {}

    def _one_sample_one_group(
        self,
        y: np.ndarray,
        known_mean: Number,
        samp_weight: Array = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
    ) -> None:

        one_sample = TaylorEstimator(parameter="mean", alpha=self.alpha)
        one_sample.estimate(
            y=y,
            samp_weight=samp_weight,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
        )

        samp_mean = one_sample.point_est
        samp_std_dev = math.sqrt(y.shape[0]) * one_sample.stderror
        samp_t_value = math.sqrt(y.shape[0]) * (samp_mean - known_mean) / samp_std_dev
        left_p_value = t.cdf(samp_t_value, y.shape[0] - 1)

        # self.design_info = {
        #     "number_strata": one_sample.number_strata,
        #     "number_psus": one_sample.number_psus,
        #     "design_effect": 0,
        #     "degrees_of_freedom": one_sample.number_psus - one_sample.number_strata,
        # }

        self.stats = {
            "number_obs": y.shape[0],
            "known_mean": known_mean,
            "df": y.shape[0] - 1,
            "t": samp_t_value,
            "p_value": {
                "less_than": left_p_value,
                "greater_than": 1 - left_p_value,
                "not_equal": 2 * t.cdf(-abs(samp_t_value), y.shape[0] - 1),
            },
        }

        self.point_est = one_sample.point_est
        self.stderror = one_sample.stderror
        self.lower_ci = one_sample.lower_ci
        self.upper_ci = one_sample.upper_ci
        self.stddev = samp_std_dev

    def _one_sample_two_groups(
        self,
        y: np.ndarray,
        group: Array = None,
        samp_weight: Array = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
    ) -> None:

        one_sample = TaylorEstimator(parameter="mean", alpha=self.alpha)
        one_sample.estimate(
            y=y,
            domain=group,
            samp_weight=samp_weight,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
        )

        group1 = one_sample.domains[0]
        group2 = one_sample.domains[1]

        mean_group1 = one_sample.point_est[group1]
        mean_group2 = one_sample.point_est[group2]

        nb_obs_group1 = np.sum(group == group1)
        nb_obs_group2 = np.sum(group == group2)

        stddev_group1 = math.sqrt(nb_obs_group1) * one_sample.stderror[group1]
        stddev_group2 = math.sqrt(nb_obs_group2) * one_sample.stderror[group2]

        t_equal_variance = (mean_group1 - mean_group2) / (
            math.sqrt(
                (
                    (nb_obs_group1 - 1) * stddev_group1 ** 2
                    + (nb_obs_group2 - 1) * stddev_group2 ** 2
                )
                / (nb_obs_group1 + nb_obs_group2 - 2)
            )
            * math.sqrt(1 / nb_obs_group1 + 1 / nb_obs_group2)
        )

        t_df_equal_variance = nb_obs_group1 + nb_obs_group2 - 2

        t_unequal_variance = (mean_group1 - mean_group2) / math.sqrt(
            stddev_group1 ** 2 / nb_obs_group1 + stddev_group2 ** 2 / nb_obs_group2
        )

        t_df_unequal_variance = math.pow(
            stddev_group1 ** 2 / nb_obs_group1 + stddev_group2 ** 2 / nb_obs_group2, 2
        ) / (
            math.pow(stddev_group1 ** 2 / nb_obs_group1, 2) / (nb_obs_group1 - 1)
            + math.pow(stddev_group2 ** 2 / nb_obs_group2, 2) / (nb_obs_group2 - 1)
        )

        left_p_value_equal_variance = t.cdf(t_equal_variance, t_df_equal_variance)
        both_p_value_equal_variance = 2 * t.cdf(-abs(t_equal_variance), t_df_equal_variance)

        left_p_value_unequal_variance = t.cdf(t_unequal_variance, t_df_unequal_variance)
        both_p_value_unequal_variance = 2 * t.cdf(-abs(t_unequal_variance), t_df_unequal_variance)

        self.stats = {
            "number_obs": {group1: nb_obs_group1, group2: nb_obs_group2},
            "t_eq_variance": t_equal_variance,
            "df_eq_variance": t_df_equal_variance,
            "t_uneq_variance": t_unequal_variance,
            "df_uneq_variance": t_df_unequal_variance,
            "p_value_eq_variance": {
                "less_than": left_p_value_equal_variance,
                "greater_than": 1 - left_p_value_equal_variance,
                "not_equal": both_p_value_equal_variance,
            },
            "p_value_uneq_variance": {
                "less_than": left_p_value_unequal_variance,
                "greater_than": 1 - left_p_value_unequal_variance,
                "not_equal": both_p_value_unequal_variance,
            },
        }

        self.point_est = one_sample.point_est
        self.stderror = one_sample.stderror
        self.lower_ci = one_sample.lower_ci
        self.upper_ci = one_sample.upper_ci
        self.stddev = {
            group1: math.sqrt(nb_obs_group1) * self.stderror[group1],
            group2: math.sqrt(nb_obs_group2) * self.stderror[group2],
        }

    def _two_samples_unpaired(
        self,
        y: np.ndarray,
        group: Array = None,
        samp_weight: Array = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
    ) -> None:

        pass

    def compare(
        self,
        y: Union[Series, Array],
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

        if self.type == "one-sample" and known_mean is not None:
            self._one_sample_one_group(
                y=y,
                known_mean=known_mean,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )
        elif self.type == "one-sample" and group is not None:
            self._one_sample_two_groups(
                y=y,
                group=group,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )
            pass
        elif self.type == "two-sample" and not self.paired:
            pass
        elif self.type == "two-sample" and self.paired:
            if len(y.shape) == 1 or y.shape[1] != 2:
                raise AssertionError(
                    "Parameter y must be an array-like object of dimension n by 2 for two-sample paired T-test"
                )
            d = y[:, 0] - y[:, 1]
            self._one_sample_one_group(
                y=d,
                known_mean=0,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )
        else:
            raise ValueError("type must be equal to 'one-sample', 'two-sample'!")

    def by(self, by_var: Array) -> Dict[StringNumber, Ttest]:
        pass