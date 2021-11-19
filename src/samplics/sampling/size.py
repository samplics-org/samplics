"""Sample size calculation module 

"""

from __future__ import annotations

import math

from typing import Optional, Union

import numpy as np
import pandas as pd

from scipy.stats import nct
from scipy.stats import norm as normal
from scipy.stats import t as student

from samplics.utils.formats import convert_numbers_to_dicts, dict_to_dataframe, numpy_array
from samplics.utils.types import Array, DictStrNum, Number, StringNumber


def calculate_power(
    two_sides: bool,
    delta: Union[Number, Array],
    sigma: Union[Number, Array],
    samp_size: Number,
    alpha: float,
):

    if isinstance(delta, (int, float)) and isinstance(sigma, (int, float)):
        if two_sides:
            return (
                1
                - normal().cdf(
                    normal().ppf(1 - alpha / 2) - delta / (sigma / math.sqrt(samp_size))
                )
                + normal().cdf(
                    -normal().ppf(1 - alpha / 2) - delta / (sigma / math.sqrt(samp_size))
                )
            )
        else:
            return 1 - normal().cdf(
                normal().ppf(1 - alpha) - delta / (sigma / math.sqrt(samp_size))
            )
    elif isinstance(delta, (np.np.ndarray, pd.Series, list, tuple)) and isinstance(
        sigma, (np.np.ndarray, pd.Series, list, tuple)
    ):
        delta = numpy_array(delta)
        sigma = numpy_array(sigma)
        power = np.zeros(delta.shape[0])
        for k in range(delta.shape[0]):
            if two_sides:
                power[k] = (
                    1
                    - normal().cdf(
                        normal().ppf(1 - alpha / 2) - delta[k] / (sigma[k] / math.sqrt(samp_size))
                    )
                    + normal().cdf(
                        -normal().ppf(1 - alpha / 2) - delta[k] / (sigma[k] / math.sqrt(samp_size))
                    )
                )
            else:
                power[k] = 1 - normal().cdf(
                    normal().ppf(1 - alpha) - delta[k] / (sigma[k] / math.sqrt(samp_size))
                )
            return power


class SampleSize:
    """*SampleSize* implements sample size calculation methods"""

    def __init__(
        self, parameter: str = "proportion", method: str = "wald", stratification: bool = False
    ) -> None:

        self.parameter = parameter.lower()
        self.method = method.lower()
        if self.parameter not in ("proportion", "mean", "total"):
            raise AssertionError("Parameter must be proportion, mean, or total.")
        if self.parameter == "proportion" and self.method not in ("wald", "fleiss"):
            raise AssertionError("For proportion, the method must be wald or Fleiss.")
        if self.parameter == "mean" and self.method not in ("wald"):
            raise AssertionError("For mean and total, the method must be wald.")

        self.stratification = stratification
        self.target: Union[DictStrNum, Number]
        self.sigma: Union[DictStrNum, Number]
        self.samp_size: Union[DictStrNum, Number] = 0
        self.deff_c: Union[DictStrNum, Number] = 1.0
        self.deff_w: Union[DictStrNum, Number] = 1.0
        self.half_ci: Union[DictStrNum, Number]
        self.resp_rate: Union[DictStrNum, Number] = 1.0
        self.pop_size: Optional[Union[DictStrNum, Number]] = None

    def icc(self) -> Union[DictStrNum, Number]:
        pass  # TODO

    def deff(
        self,
        cluster_size: Union[DictStrNum, Number],
        icc: Union[DictStrNum, Number],
    ) -> Union[DictStrNum, Number]:

        if isinstance(cluster_size, (int, float)) and isinstance(icc, (int, float)):
            return max(1 + (cluster_size - 1) * icc, 0)
        elif isinstance(cluster_size, dict) and isinstance(icc, dict):
            if cluster_size.keys() != icc.keys():
                raise AssertionError("Parameters do not have the same dictionary keys.")
            deff_c: DictStrNum = {}
            for s in cluster_size:
                deff_c[s] = max(1 + (cluster_size[s] - 1) * icc[s], 0)
            return deff_c
        else:
            raise ValueError("Combination of types not supported.")

    @staticmethod
    def _calculate_ss_prop_wald(
        target: Union[DictStrNum, Number],
        half_ci: Union[DictStrNum, Number],
        pop_size: Optional[Union[DictStrNum, Number]],
        deff_c: Union[DictStrNum, Number],
        alpha: float,
    ) -> Union[DictStrNum, Number]:

        z_value = normal().ppf(1 - alpha / 2)

        if isinstance(target, dict) and isinstance(half_ci, dict) and isinstance(deff_c, dict):
            samp_size: DictStrNum = {}
            for s in half_ci:
                sigma_s = target[s] * (1 - target[s])
                if isinstance(pop_size, dict):
                    samp_size[s] = math.ceil(
                        deff_c[s]
                        * pop_size[s]
                        * z_value ** 2
                        * sigma_s
                        / ((pop_size[s] - 1) * half_ci[s] ** 2 + z_value * sigma_s)
                    )
                else:
                    samp_size[s] = math.ceil(deff_c[s] * z_value ** 2 * sigma_s / half_ci[s] ** 2)
            return samp_size
        elif (
            isinstance(target, (int, float))
            and isinstance(half_ci, (int, float))
            and isinstance(deff_c, (int, float))
        ):
            sigma = target * (1 - target)
            if isinstance(pop_size, (int, float)):
                return math.ceil(
                    deff_c
                    * pop_size
                    * z_value ** 2
                    * sigma
                    / ((pop_size - 1) * half_ci ** 2 + z_value * sigma)
                )
            else:
                return math.ceil(deff_c * z_value ** 2 * sigma / half_ci ** 2)
        else:
            raise TypeError("target and half_ci must be numbers or dictionaries!")

    @staticmethod
    def _calculate_ss_prop_fleiss(
        target: Union[DictStrNum, Number],
        half_ci: Union[DictStrNum, Number],
        # pop_size: Optional[Union[DictStrNum, Number]],
        deff_c: Union[DictStrNum, Number],
        alpha: float,
    ) -> Union[DictStrNum, Number]:

        z_value = normal().ppf(1 - alpha / 2)

        def fleiss_factor(p: float, d: float) -> float:

            if 0 <= p < d or 1 - d < p <= 1:
                return 8 * d * (1 - 2 * d)
            elif d <= p < 0.3:
                return 4 * (p + d) * (1 - p - d)
            elif 0.7 < p <= 1 - d:
                return 4 * (p - d) * (1 - p + d)
            elif 0.3 <= p <= 0.7:
                return 1
            else:
                raise ValueError("Parameters p or d not valid.")

        if isinstance(target, dict) and isinstance(half_ci, dict) and isinstance(deff_c, dict):
            samp_size: DictStrNum = {}
            for s in half_ci:
                fct = fleiss_factor(target[s], half_ci[s])
                samp_size[s] = math.ceil(
                    deff_c[s]
                    * (
                        fct * (z_value ** 2) / (4 * half_ci[s] ** 2)
                        + 1 / half_ci[s]
                        - 2 * z_value ** 2
                        + (z_value + 2) / fct
                    )
                )
            return samp_size
        elif (
            isinstance(target, (int, float))
            and isinstance(half_ci, (int, float))
            and isinstance(deff_c, (int, float))
        ):
            fct = fleiss_factor(target, half_ci)
            return math.ceil(
                deff_c
                * (
                    fct * (z_value ** 2) / (4 * half_ci ** 2)
                    + 1 / half_ci
                    - 2 * z_value ** 2
                    + (z_value + 2) / fct
                )
            )
        else:
            raise TypeError("target and half_ci must be numbers or dictionaries!")

    @staticmethod
    def _calculate_ss_mean_wald(
        half_ci: Union[DictStrNum, Number],
        sigma: Union[DictStrNum, Number],
        pop_size: Optional[Union[DictStrNum, Number]],
        deff_c: Union[DictStrNum, Number],
        alpha: float,
    ) -> Union[DictStrNum, Number]:

        z_value = normal().ppf(1 - alpha / 2)

        if isinstance(half_ci, dict) and isinstance(sigma, dict) and isinstance(deff_c, dict):
            samp_size: DictStrNum = {}
            for s in half_ci:
                if isinstance(pop_size, dict):
                    samp_size[s] = math.ceil(
                        deff_c[s]
                        * pop_size[s]
                        * z_value ** 2
                        * sigma[s] ** 2
                        / ((pop_size[s] - 1) * half_ci[s] ** 2 + z_value ** 2 * sigma[s] ** 2)
                    )
                else:
                    samp_size[s] = math.ceil(deff_c[s] * (z_value * sigma[s] / half_ci[s]) ** 2)
            return samp_size
        elif (
            isinstance(half_ci, (int, float))
            and isinstance(sigma, (int, float))
            and isinstance(deff_c, (int, float))
        ):
            if isinstance(pop_size, (int, float)):
                return math.ceil(
                    deff_c
                    * pop_size
                    * z_value ** 2
                    * sigma ** 2
                    / ((pop_size - 1) * half_ci ** 2 + z_value ** 2 * sigma ** 2)
                )
            else:
                return math.ceil(deff_c * (z_value * sigma / half_ci) ** 2)
        else:
            raise TypeError("target, half_ci, and sigma must be numbers or dictionaries!")

    def calculate(
        self,
        target: Union[DictStrNum, Number],
        half_ci: Union[DictStrNum, Number],
        sigma: Optional[Union[DictStrNum, Number]] = None,
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
    ) -> None:

        is_target_dict = isinstance(target, dict)
        is_sigma_dict = isinstance(sigma, dict)
        is_half_ci_dict = isinstance(half_ci, dict)
        is_deff_dict = isinstance(deff, dict)
        is_resp_rate_dict = isinstance(resp_rate, dict)
        is_pop_size_dict = isinstance(pop_size, dict)

        number_dictionaries = (
            is_target_dict
            + is_sigma_dict
            + is_half_ci_dict
            + is_deff_dict
            + is_resp_rate_dict
            + is_pop_size_dict
        )

        if self.parameter == "proportion" and target is None:
            raise AssertionError(
                "target must be provided to calculate sample size for proportion."
            )

        if self.parameter == "mean" and sigma is None:
            raise AssertionError("sigma must be provided to calculate sample size for mean.")

        if self.parameter == "proportion":
            if isinstance(target, (int, float)) and not 0 <= target <= 1:
                raise ValueError("Target for proportions must be between 0 and 1.")
            if isinstance(target, dict):
                for s in target:
                    if not 0 <= target[s] <= 1:
                        raise ValueError("Target for proportions must be between 0 and 1.")

        if self.parameter == "proportion" and sigma is None:
            if isinstance(target, (int, float)):
                sigma = target * (1 - target)
            if isinstance(target, dict):
                sigma = {}
                for s in target:
                    sigma[s] = target[s] * (1 - target[s])

        stratum: Optional[list[StringNumber]] = None
        if not self.stratification and (
            isinstance(half_ci, dict)
            or isinstance(target, dict)
            or isinstance(sigma, dict)
            or isinstance(deff, dict)
            or isinstance(resp_rate, dict)
            or isinstance(pop_size, dict)
        ):
            raise AssertionError("No python dictionary needed for non-stratified sample.")
        elif (
            not self.stratification
            and isinstance(half_ci, (int, float))
            and isinstance(target, (int, float))
            and isinstance(sigma, (int, float))
            and isinstance(deff, (int, float))
            and isinstance(resp_rate, (int, float))
        ):
            self.half_ci = half_ci
            self.target = target
            self.sigma = sigma
            self.deff_c = deff
            self.resp_rate = resp_rate
        elif (
            self.stratification
            and isinstance(half_ci, (int, float))
            and isinstance(target, (int, float))
            and isinstance(sigma, (int, float))
            and isinstance(deff, (int, float))
            and isinstance(resp_rate, (int, float))
        ):
            if number_strata is not None:
                stratum = ["_stratum_" + str(i) for i in range(1, number_strata + 1)]
                self.half_ci = dict(zip(stratum, np.repeat(half_ci, number_strata)))
                self.target = dict(zip(stratum, np.repeat(target, number_strata)))
                self.sigma = dict(zip(stratum, np.repeat(sigma, number_strata)))
                self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
                self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
                if isinstance(pop_size, (int, float)):
                    self.pop_size = dict(zip(stratum, np.repeat(pop_size, number_strata)))
            else:
                raise ValueError("Number of strata not specified!")
        elif self.stratification and number_dictionaries > 0:
            dict_number = 0
            for ll in [target, half_ci, sigma, deff, resp_rate, pop_size]:
                if isinstance(ll, dict):
                    dict_number += 1
                    if dict_number == 1:
                        stratum = list(ll.keys())
                    elif dict_number > 0:
                        if stratum != list(ll.keys()):
                            raise AssertionError("Python dictionaries have different keys")
            number_strata = len(stratum) if stratum is not None else 0
            if not is_target_dict and isinstance(target, (int, float)) and stratum is not None:
                self.target = dict(zip(stratum, np.repeat(target, number_strata)))
            elif isinstance(target, dict):
                self.target = target
            if not is_sigma_dict and isinstance(sigma, (int, float)) and stratum is not None:
                self.sigma = dict(zip(stratum, np.repeat(sigma, number_strata)))
            elif isinstance(sigma, dict):
                self.sigma = sigma
            if not is_half_ci_dict and isinstance(half_ci, (int, float)) and stratum is not None:
                self.half_ci = dict(zip(stratum, np.repeat(half_ci, number_strata)))
            elif isinstance(half_ci, dict):
                self.half_ci = half_ci
            if not is_deff_dict and isinstance(deff, (int, float)) and stratum is not None:
                self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
            elif isinstance(deff, dict):
                self.deff_c = deff
            if (
                not is_resp_rate_dict
                and isinstance(resp_rate, (int, float))
                and stratum is not None
            ):
                self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
            elif isinstance(resp_rate, dict):
                self.resp_rate = resp_rate
            if (
                not isinstance(pop_size, dict)
                and isinstance(pop_size, (int, float))
                and stratum is not None
            ):
                self.pop_size = dict(zip(stratum, np.repeat(pop_size, number_strata)))
            elif isinstance(pop_size, dict):
                self.pop_size = pop_size

        self.alpha = alpha

        samp_size: Union[DictStrNum, Number]
        if self.parameter == "proportion" and self.method == "wald":
            samp_size = self._calculate_ss_prop_wald(
                half_ci=self.half_ci,
                target=self.target,
                pop_size=self.pop_size,
                deff_c=self.deff_c,
                alpha=self.alpha,
            )
        elif self.parameter == "proportion" and self.method == "fleiss":
            samp_size = self._calculate_ss_prop_fleiss(
                half_ci=self.half_ci,
                target=self.target,
                # pop_size=self.pop_size,
                deff_c=self.deff_c,
                alpha=self.alpha,
            )
        elif self.parameter in ("mean", "total") and self.method == "wald":
            samp_size = self._calculate_ss_mean_wald(
                half_ci=self.half_ci,
                sigma=self.sigma,
                pop_size=self.pop_size,
                deff_c=self.deff_c,
                alpha=self.alpha,
            )

        self.samp_size = samp_size

    def to_dataframe(self, col_names: Optional[list[str]] = None) -> pd.DataFrame:
        """Coverts the dictionaries to a pandas dataframe

        Args:
            col_names (list[str], optional): column names for the dataframe. Defaults to
                ["_stratum", "_target", "_half_ci", "_samp_size"].

        Raises:
            AssertionError: when sample size is not calculated.

        Returns:
            pd.DataFrame: output pandas dataframe.
        """

        if self.samp_size is None:
            raise AssertionError("No sample size calculated.")
        elif col_names is None:
            col_names = [
                "_parameter",
                "_stratum",
                "_target",
                "_sigma",
                "_half_ci",
                "_samp_size",
            ]
            if not self.stratification:
                col_names.pop(1)
        else:
            ncols = len(col_names)
            if (ncols != 6 and self.stratification) or (ncols != 5 and not self.stratification):
                raise AssertionError(
                    "col_names must have 6 values for stratified design and 5 for not stratified design."
                )
        est_df = dict_to_dataframe(
            col_names,
            self.target,
            self.sigma,
            self.half_ci,
            self.samp_size,
        )
        est_df.iloc[:, 0] = self.parameter

        return est_df


def allocate(
    method: str,
    stratum: Array,
    pop_size: DictStrNum,
    samp_size: Optional[Number] = None,
    constant: Optional[Number] = None,
    rate: Optional[Union[DictStrNum, Number]] = None,
    stddev: Optional[DictStrNum] = None,
) -> tuple[DictStrNum, DictStrNum]:
    """Reference: Kish(1965), page 97"""

    stratum = list(numpy_array(stratum))

    if method.lower() == "equal":
        if isinstance(constant, (int, float)):
            sample_sizes = dict(zip(stratum, np.repeat(constant, len(stratum))))
        else:
            raise ValueError("Parameter 'target_size' must be a valid integer!")
    elif method.lower() == "proportional":
        if isinstance(pop_size, dict) and stddev is None and samp_size is not None:
            total_pop = sum(list(pop_size.values()))
            samp_size_h = [math.ceil((samp_size / total_pop) * pop_size[k]) for k in stratum]
            sample_sizes = dict(zip(stratum, samp_size_h))
        elif isinstance(pop_size, dict) and stddev is not None and samp_size is not None:
            total_pop = sum(list(pop_size.values()))
            samp_size_h = [
                math.ceil((samp_size / total_pop) * pop_size[k] * stddev[k]) for k in stratum
            ]
            sample_sizes = dict(zip(stratum, samp_size_h))
        else:
            raise ValueError(
                "Parameter 'pop_size' must be a dictionary and 'samp_size' an integer!"
            )
    elif method.lower() == "fixed_rate":
        if isinstance(rate, (int, float)) and pop_size is not None:
            samp_size_h = [math.ceil(rate * pop_size[k]) for k in stratum]
        else:
            raise ValueError(
                "Parameter 'pop_size' and 'rate' must be a dictionary and number respectively!"
            )
        sample_sizes = dict(zip(stratum, samp_size_h))
    elif method.lower() == "proportional_rate":
        if isinstance(rate, (int, float)) and pop_size is not None:
            samp_size_h = [math.ceil(rate * pop_size[k] * pop_size[k]) for k in stratum]
        else:
            raise ValueError("Parameter 'pop_size' must be a dictionary!")
        sample_sizes = dict(zip(stratum, samp_size_h))
    elif method.lower() == "equal_errors":
        if isinstance(constant, (int, float)) and stddev is not None:
            samp_size_h = [math.ceil(constant * stddev[k] * stddev[k]) for k in stratum]
        else:
            raise ValueError(
                "Parameter 'stddev' and 'constant' must be a dictionary and number, respectively!"
            )
        sample_sizes = dict(zip(stratum, samp_size_h))
    elif method.lower() == "optimum_mean":
        if isinstance(rate, (int, float)) and pop_size is not None and stddev is not None:
            samp_size_h = [math.ceil(rate * pop_size[k] * stddev[k]) for k in stratum]
        else:
            raise ValueError(
                "Parameter 'pop_size' and 'rate' must be a dictionary and number respectively!"
            )
        sample_sizes = dict(zip(stratum, samp_size_h))
    elif method.lower() == "optimum_comparison":
        if isinstance(rate, (int, float)) and stddev is not None:
            samp_size_h = [math.ceil(rate * stddev[k]) for k in stratum]
        else:
            raise ValueError(
                "Parameter 'stddev' and 'rate' must be a dictionary and number respectively!"
            )
        sample_sizes = dict(zip(stratum, samp_size_h))
    elif method.lower() == "variable_rate" and isinstance(rate, dict) and pop_size is not None:
        samp_size_h = [math.ceil(rate[k] * pop_size[k]) for k in stratum]
        sample_sizes = dict(zip(stratum, samp_size_h))
    else:
        raise ValueError(
            "Parameter 'method' is not valid. Options are 'equal', 'proportional', 'fixed_rate', 'proportional_rate', 'equal_errors', 'optimun_mean', and 'optimun_comparison'!"
        )

    if (list(sample_sizes.values()) > np.asarray(list(pop_size.values()))).any():
        raise ValueError(
            "Some of the constants, rates, or standard errors are too large, resulting in sample sizes larger than population sizes!"
        )

    sample_rates = {}
    if isinstance(pop_size, dict):
        for k in pop_size:
            sample_rates[k] = sample_sizes[k] / pop_size[k]

    return sample_sizes, sample_rates


def calculate_clusters() -> None:
    pass


class SampleSizeForDifference:
    """SampleSizeHypothesisTesting implements sample size calculation when the objective is to compare groups or against a target"""

    def __init__(
        self,
        parameter: str = "proportion",
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        self.parameter = parameter.lower()
        self.method = method.lower()
        if self.parameter not in ("proportion", "mean", "total"):
            raise AssertionError("Parameter must be proportion, mean, total.")
        if self.method not in ("wald"):
            raise AssertionError("The method must be wald.")

        self.stratification = stratification
        self.two_sides = two_sides
        self.params_estimated = params_estimated

        self.alpha = Number
        self.beta = Number
        self.samp_size: Union[DictStrNum, Number]
        self.power: Union[DictStrNum, Number]
        self.actual_power: Union[DictStrNum, Number]
        self.delta: Union[DictStrNum, Number]
        self.sigma: Union[DictStrNum, Number]
        self.deff_c: Union[DictStrNum, Number]
        self.deff_w: Union[DictStrNum, Number]
        self.resp_rate: Union[DictStrNum, Number]
        self.pop_size: Optional[Union[DictStrNum, Number]] = None

    def _input_parameters_validation(
        self,
        delta: Union[DictStrNum, Number],
        sigma: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        # pop_size: Optional[Union[DictStrNum, Number]] = None,
    ) -> None:

        is_delta_dict = isinstance(delta, dict)
        is_sigma_dict = isinstance(sigma, dict)
        is_deff_dict = isinstance(deff, dict)
        is_resp_rate_dict = isinstance(resp_rate, dict)
        # is_pop_size_dict = isinstance(pop_size, dict)

        number_dictionaries = (
            is_delta_dict + is_sigma_dict + is_deff_dict + is_resp_rate_dict  # + is_pop_size_dict
        )

        if self.parameter == "proportion":
            if isinstance(delta, (int, float)) and not 0 <= delta <= 1:
                raise ValueError("Target for proportions must be between 0 and 1.")
            if isinstance(delta, dict):
                for s in delta:
                    if not 0 <= delta[s] <= 1:
                        raise ValueError("Target for proportions must be between 0 and 1.")

        stratum: Optional[list[StringNumber]] = None
        if not self.stratification and (
            isinstance(delta, dict)
            or isinstance(sigma, dict)
            or isinstance(deff, dict)
            or isinstance(resp_rate, dict)
            # or isinstance(pop_size, dict)
        ):
            raise AssertionError("No python dictionary needed for non-stratified sample.")
        elif (
            not self.stratification
            and isinstance(delta, (int, float))
            and isinstance(sigma, (int, float))
            and isinstance(deff, (int, float))
            and isinstance(resp_rate, (int, float))
        ):
            self.delta = delta
            self.sigma = sigma
            self.deff_c = deff
            self.resp_rate = resp_rate
        elif (
            self.stratification
            and isinstance(delta, (int, float))
            and isinstance(sigma, (int, float))
            and isinstance(deff, (int, float))
            and isinstance(resp_rate, (int, float))
        ):
            if number_strata is not None:
                stratum = ["_stratum_" + str(i) for i in range(1, number_strata + 1)]
                self.target = dict(zip(stratum, np.repeat(delta, number_strata)))
                self.sigma = dict(zip(stratum, np.repeat(sigma, number_strata)))
                self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
                self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
                # if isinstance(pop_size, (int, float)):
                #     self.pop_size = dict(zip(stratum, np.repeat(pop_size, number_strata)))
            else:
                raise ValueError("Number of strata not specified!")
        elif self.stratification and number_dictionaries > 0:
            dict_number = 0
            for ll in [delta, sigma, deff, resp_rate]:  # , pop_size]:
                if isinstance(ll, dict):
                    dict_number += 1
                    if dict_number == 1:
                        stratum = list(ll.keys())
                    elif dict_number > 0:
                        if stratum != list(ll.keys()):
                            raise AssertionError("Python dictionaries have different keys")
            number_strata = len(stratum) if stratum is not None else 0
            if not is_delta_dict and isinstance(delta, (int, float)) and stratum is not None:
                self.delta = dict(zip(stratum, np.repeat(delta, number_strata)))
            elif isinstance(delta, dict):
                self.delta = delta
            if not is_sigma_dict and isinstance(sigma, (int, float)) and stratum is not None:
                self.sigma = dict(zip(stratum, np.repeat(sigma, number_strata)))
            elif isinstance(sigma, dict):
                self.sigma = sigma
            if not is_deff_dict and isinstance(deff, (int, float)) and stratum is not None:
                self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
            elif isinstance(deff, dict):
                self.deff_c = deff
            if (
                not is_resp_rate_dict
                and isinstance(resp_rate, (int, float))
                and stratum is not None
            ):
                self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
            elif isinstance(resp_rate, dict):
                self.resp_rate = resp_rate
            # if (
            #     not isinstance(pop_size, dict)
            #     and isinstance(pop_size, (int, float))
            #     and stratum is not None
            # ):
            #     self.pop_size = dict(zip(stratum, np.repeat(pop_size, number_strata)))
            # elif isinstance(pop_size, dict):
            #     self.pop_size = pop_size

    @staticmethod
    def _calculate_ss_difference_mean_wald(
        two_sides: bool,
        delta: Union[DictStrNum, Number],
        sigma: Union[DictStrNum, Number],
        deff_c: Union[DictStrNum, Number],
        alpha: float,
        power: float,
    ) -> Union[DictStrNum, Number]:

        if two_sides:
            z_alpha = normal().ppf(1 - alpha / 2)
        else:
            z_alpha = normal().ppf(1 - alpha)
        z_beta = normal().ppf(power)

        if isinstance(delta, dict) and isinstance(sigma, dict) and isinstance(deff_c, dict):
            samp_size: DictStrNum = {}
            for s in delta:
                samp_size[s] = math.ceil(deff_c[s] * ((z_alpha + z_beta) * sigma[s] / delta) ** 2)
            return samp_size
        elif (
            isinstance(delta, (int, float))
            and isinstance(sigma, (int, float))
            and isinstance(deff_c, (int, float))
        ):
            return math.ceil(deff_c * ((z_alpha + z_beta) * sigma / delta) ** 2)
        else:
            raise TypeError("target, half_ci, and sigma must be numbers or dictionaries!")

    def calculate(
        self,
        delta: Union[DictStrNum, Number],
        sigma: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        # pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> None:

        self._input_parameters_validation(delta, sigma, deff, resp_rate, number_strata)

        self.alpha = alpha
        self.power = power

        # samp_size: Union[DictStrNum, Number]
        if self.parameter in ("proportion", "mean", "total") and self.method == "wald":
            self.samp_size = self._calculate_ss_difference_mean_wald(
                two_sides=self.two_sides,
                delta=self.delta,
                sigma=self.sigma,
                deff_c=self.deff_c,
                alpha=self.alpha,
                power=self.power,
            )

        # self.samp_size = samp_size

        if self.stratification:
            for k in self.samp_size:
                self.actual_power[k] = calculate_power(
                    self.two_sides, self.delta[k], self.sigma[k], self.samp_size[k], self.alpha
                )
        else:
            self.actual_power = calculate_power(
                self.two_sides, self.delta, self.sigma, self.samp_size, self.alpha
            )


class SampleSizeOneGroup:
    """SampleSizeHypothesisTesting implements sample size calculation when the objective is to compare groups or against a target"""

    def __init__(
        self,
        parameter: str = "proportion",
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        self.parameter = parameter.lower()
        self.method = method.lower()
        if self.parameter not in ("proportion", "mean", "total"):
            raise AssertionError("Parameter must be proportion, mean, total.")
        if self.method not in ("wald"):
            raise AssertionError("The method must be wald.")

        self.stratification = stratification
        self.two_sides = two_sides
        self.params_estimated = params_estimated

        self.alpha = Number
        self.beta = Number
        self.samp_size: Union[DictStrNum, Number]
        self.power: Union[DictStrNum, Number]
        self.target_0: Union[DictStrNum, Number]
        self.target_1: Union[DictStrNum, Number]
        self.stddev: Union[DictStrNum, Number]
        self.deff_c: Union[DictStrNum, Number]
        self.deff_w: Union[DictStrNum, Number]
        self.resp_rate: Union[DictStrNum, Number]
        self.pop_size: Optional[Union[DictStrNum, Number]] = None

    def calculate(
        self,
        target_0: Union[DictStrNum, Number],
        target_1: Union[DictStrNum, Number],
        sigma: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> None:

        delta: Union[DictStrNum, Number]
        if isinstance(target_0, (int, float)) and isinstance(target_1, (int, float)):
            delta = target_1 - target_0
        elif isinstance(target_0, dict) and isinstance(target_1, dict):
            delta = {k: target_1[k] - target_0[k] for k in target_0}
        else:
            raise AssertionError("target_0 and targget_1 are not the same type.")

        SampSizeDiff = SampleSizeForDifference(
            parameter=self.parameter,
            method=self.method,
            stratification=self.stratification,
            two_sides=self.two_sides,
            params_estimated=self.params_estimated,
        )

        SampSizeDiff.calculate(
            delta=delta, sigma=sigma, deff=deff, resp_rate=resp_rate, alpha=alpha, power=power
        )


class SampleSizeTwoGroups:
    """SampleSizeHypothesisTesting implements sample size calculation when the objective is to compare groups or against a target"""

    def __init__(
        self,
        parameter: str = "proportion",
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        self.stratification = stratification
        self.two_sides = two_sides
        self.params_estimated = params_estimated

        self.alpha = Number
        self.beta = Number
        self.samp_size: Union[DictStrNum, Number]
        self.power: Union[DictStrNum, Number]
        self.delta: Union[DictStrNum, Number]
        self.param_a: Union[DictStrNum, Number]
        self.stddev_a: Union[DictStrNum, Number]
        self.param_b: Union[DictStrNum, Number]
        self.stddev_a: Union[DictStrNum, Number]
        self.stddev_b: Union[DictStrNum, Number]
        self.deff_c: Union[DictStrNum, Number]
        self.deff_w: Union[DictStrNum, Number]
        self.resp_rate: Union[DictStrNum, Number]
        self.pop_size: Optional[Union[DictStrNum, Number]] = None

    def calculate(
        self,
        delta: Union[DictStrNum, Number],
        param_a: Union[DictStrNum, Number],
        param_b: Union[DictStrNum, Number],
        stddev_a: Union[DictStrNum, Number],
        stddev_b: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> None:
        pass


class OneMeanSampleSize:
    """*OneMeanSample* implements sample size calculation methods for a mean for one single population."""

    def __init__(
        self, stratification: bool = False, two_side: bool = True, estimated_mean: bool = True
    ) -> None:

        self.stratification = stratification
        self.test_type = "two-side" if two_side else "one-side"
        self.estimated_mean = estimated_mean

        self.alpha = Number
        self.beta = Number
        self.samp_size: Union[DictStrNum, Number] = 0
        self.power: Union[DictStrNum, Number] = 0
        self.targeted_mean: Union[DictStrNum, Number]
        self.reference_mean: Union[DictStrNum, Number]
        self.stddev: Union[DictStrNum, Number]
        self.deff_c: Union[DictStrNum, Number] = 1.0
        self.deff_w: Union[DictStrNum, Number] = 1.0
        self.resp_rate: Union[DictStrNum, Number] = 1.0
        self.pop_size: Optional[Union[DictStrNum, Number]] = None

    def calculate(
        self,
        targeted_mean: Union[DictStrNum, Number],
        reference_mean: Union[DictStrNum, Number],
        stddev: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
        beta: float = 0.20,
    ) -> None:

        is_targeted_mean_dict = isinstance(targeted_mean, dict)
        is_reference_mean_dict = isinstance(reference_mean, dict)
        is_stddev_dict = isinstance(stddev, dict)
        is_deff_dict = isinstance(deff, dict)
        is_resp_rate_dict = isinstance(resp_rate, dict)
        is_pop_size_dict = isinstance(pop_size, dict)

        number_dictionaries = (
            is_targeted_mean_dict
            + is_reference_mean_dict
            + is_stddev_dict
            + is_deff_dict
            + is_resp_rate_dict
            + is_pop_size_dict
        )

        if (
            self.stratification
            and (number_strata is None or number_strata <= 0)
            and number_dictionaries == 0
        ):
            raise AssertionError("Stratified designs ")

        if not self.stratification and number_dictionaries >= 1:
            raise ValueError("Dictionaries must NOT be provided for non stratified designs!")

        number_strata = 1 if not self.stratification else number_strata

        [mean1, mean0, std_dev] = convert_numbers_to_dicts(
            number_strata, targeted_mean, reference_mean, stddev
        )

        self.alpha = alpha
        self.beta = beta

        prob_alpha = (
            normal.ppf(1 - self.alpha / 2)
            if self.test_type == "two-side"
            else normal.ppf(1 - self.alpha)
        )
        prob_beta = normal.ppf(1 - self.beta)
        if self.stratification:
            self.samp_size = {}
            self.power = {}
            for key in mean1:
                self.samp_size[key] = math.ceil(
                    pow(
                        (prob_alpha + prob_beta) * std_dev[key] / (mean1[key] - mean0[key]),
                        2,
                    )
                )

                if self.estimated_mean:
                    t_prob_alpha = (
                        student.ppf(1 - self.alpha / 2, self.samp_size[key] - 1)
                        if self.test_type == "two-side"
                        else student.ppf(1 - self.alpha, self.samp_size[key] - 1)
                    )
                    t_prob_beta = student.ppf(1 - self.beta, self.samp_size[key] - 1)
                    self.samp_size[key] = math.ceil(
                        pow(
                            (t_prob_alpha + t_prob_beta)
                            * std_dev[key]
                            / (mean1[key] - mean0[key]),
                            2,
                        )
                    )

                adj_fct = (mean0[key] - mean1[key]) / (
                    std_dev[key] / math.sqrt(self.samp_size[key])
                )
                self.power[key] = (
                    1 - normal.cdf(prob_alpha + adj_fct) + normal.cdf(-prob_alpha + adj_fct)
                )
        else:
            self.samp_size = math.ceil(
                pow(
                    (prob_alpha + prob_beta)
                    * std_dev["_stratum_1"]
                    / (mean1["_stratum_1"] - mean0["_stratum_1"]),
                    2,
                )
            )

            adj_fct = (mean0["_stratum_1"] - mean1["_stratum_1"]) / (
                std_dev["_stratum_1"] / math.sqrt(self.samp_size)
            )
            self.power = 1 - normal.cdf(prob_alpha + adj_fct) + normal.cdf(-prob_alpha + adj_fct)

            if self.estimated_mean:
                t_prob_alpha0 = (
                    student.ppf(1 - self.alpha / 2, self.samp_size - 1)
                    if self.test_type == "two-side"
                    else student.ppf(1 - self.alpha, self.samp_size - 1)
                )
                t_prob_beta0 = student.ppf(1 - self.beta, self.samp_size - 1)
                self.samp_size = math.ceil(
                    pow(
                        (t_prob_alpha0 + t_prob_beta0)
                        * std_dev["_stratum_1"]
                        / (mean1["_stratum_1"] - mean0["_stratum_1"]),
                        2,
                    )
                )
                t_prob_alpha1 = (
                    student.ppf(1 - self.alpha / 2, self.samp_size - 1)
                    if self.test_type == "two-side"
                    else student.ppf(1 - self.alpha, self.samp_size - 1)
                )

                t_adj_fct = (mean0["_stratum_1"] - mean1["_stratum_1"]) / (
                    std_dev["_stratum_1"] / math.sqrt(self.samp_size)
                )
                self.power = (
                    1
                    - nct.cdf(t_prob_alpha1, self.samp_size - 1, -t_adj_fct)
                    # + student.cdf(-t_prob_alpha1 + t_adj_fct)
                )
