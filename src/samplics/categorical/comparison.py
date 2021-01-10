"""Comparison module 

The module implements comparisons of groups.

"""

from __future__ import annotations
from typing import Any, Generic, Dict, List, Optional, Union, Tuple

import math
import numpy as np
import pandas as pd

from scipy.stats import t

from samplics.utils.checks import assert_probabilities
from samplics.utils.formats import numpy_array, remove_nans
from samplics.utils.types import Array, Number, Series, StringNumber

from samplics.estimation import TaylorEstimator


class Ttest(Generic[Number, StringNumber]):
    def __init__(self, samp_type: str, paired: bool = False, alpha: float = 0.05) -> None:

        if samp_type.lower() not in ("one-sample", "two-sample"):
            raise ValueError("Parameter 'type' must be equal to 'one-sample', 'two-sample'!")
        assert_probabilities(alpha)

        self.samp_type = samp_type.lower()
        self.paired = paired

        self.point_est: Any = {}
        self.stats: Any = {}
        self.stderror: Any = {}
        self.stddev: Any = {}
        self.lower_ci: Any = {}
        self.upper_ci: Any = {}
        self.deff: Any = {}
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

    def _two_groups_unpaired(
        self, mean_est: TaylorEstimator, group: np.ndarray
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:

        if self.samp_type == "one-sample":
            group1 = mean_est.domains[0]
            group2 = mean_est.domains[1]
        elif self.samp_type == "two-sample":
            group1 = mean_est.by[0]
            group2 = mean_est.by[1]
        else:
            raise ValueError("Parameter type is not valid!")

        mean_group1 = mean_est.point_est[group1]
        mean_group2 = mean_est.point_est[group2]

        nb_obs_group1 = np.sum(group == group1)
        nb_obs_group2 = np.sum(group == group2)

        stddev_group1 = math.sqrt(nb_obs_group1) * mean_est.stderror[group1]
        stddev_group2 = math.sqrt(nb_obs_group2) * mean_est.stderror[group2]

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

        stats = {
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

        if (
            isinstance(mean_est.point_est, dict)
            and isinstance(mean_est.stderror, dict)
            and isinstance(mean_est.lower_ci, dict)
            and isinstance(mean_est.upper_ci, dict)
        ):
            point_est = mean_est.point_est.copy()
            stderror = mean_est.stderror.copy()
            lower_ci = mean_est.lower_ci.copy()
            upper_ci = mean_est.upper_ci.copy()
        else:
            point_est = mean_est.point_est
            stderror = mean_est.stderror
            lower_ci = mean_est.lower_ci
            upper_ci = mean_est.upper_ci

        stddev = {
            group1: math.sqrt(nb_obs_group1) * stderror[group1],
            group2: math.sqrt(nb_obs_group2) * stderror[group2],
        }

        return point_est, stderror, stddev, lower_ci, upper_ci, stats

    # def _one_sample_two_groups(
    #     self,
    #     y: np.ndarray,
    #     group: Array = None,
    #     samp_weight: Array = None,
    #     stratum: Optional[Array] = None,
    #     psu: Optional[Array] = None,
    #     ssu: Optional[Array] = None,
    #     fpc: Union[Dict, float] = 1,
    # ) -> None:

    #     one_sample = TaylorEstimator(parameter="mean", alpha=self.alpha)
    #     one_sample.estimate(
    #         y=y,
    #         domain=group,
    #         samp_weight=samp_weight,
    #         stratum=stratum,
    #         psu=psu,
    #         ssu=ssu,
    #         fpc=fpc,
    #     )
    #     self._two_groups_unpaired(mean_est=one_sample, group=group)

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

        two_samples_unpaired = TaylorEstimator(parameter="mean", alpha=self.alpha)
        two_samples_unpaired.estimate(
            y=y,
            by=group,
            samp_weight=samp_weight,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
        )
        self._two_groups_unpaired(mean_est=two_samples_unpaired, group=group)

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

        if self.samp_type == "one-sample" and known_mean is not None:
            self._one_sample_one_group(
                y=y,
                known_mean=known_mean,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )
        elif self.samp_type == "one-sample" and group is not None:
            # self._one_sample_two_groups(
            #     y=y,
            #     group=group,
            #     samp_weight=samp_weight,
            #     stratum=stratum,
            #     psu=psu,
            #     ssu=ssu,
            #     fpc=fpc,
            # )
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
            (
                self.point_est,
                self.stderror,
                self.stddev,
                self.lower_ci,
                self.upper_ci,
                self.stats,
            ) = self._two_groups_unpaired(mean_est=one_sample, group=group)
        elif self.samp_type == "two-sample" and not self.paired:
            self._two_samples_unpaired(
                y=y,
                group=group,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )

            two_samples_unpaired = TaylorEstimator(parameter="mean", alpha=self.alpha)
            two_samples_unpaired.estimate(
                y=y,
                by=group,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )
            (
                self.point_est,
                self.stderror,
                self.stddev,
                self.lower_ci,
                self.upper_ci,
                self.stats,
            ) = self._two_groups_unpaired(mean_est=two_samples_unpaired, group=group)
        elif self.samp_type == "two-sample" and self.paired:
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

    # def by(self, by_var: Array) -> Dict[StringNumber, Ttest]:
    #     pass