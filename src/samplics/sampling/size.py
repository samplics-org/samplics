"""Sample size calculation module 

"""

from __future__ import annotations

import math

from typing import Optional, Union

import numpy as np
import pandas as pd

from scipy.stats import norm as normal
from scipy.stats import t as student

from samplics.utils.formats import convert_numbers_to_dicts, dict_to_dataframe, numpy_array
from samplics.utils.types import Array, DictStrNum, Number, StringNumber


def power_for_proportion(
    two_sides: bool,
    prop_0: Union[DictStrNum, Number, Array],
    prop_1: Union[DictStrNum, Number, Array],
    samp_size: Union[DictStrNum, Number, Array],
    arcsin: bool = True,
    alpha: float = 0.05,
):

    z_value = normal().ppf(1 - alpha / 2)

    if isinstance(prop_0, dict) and isinstance(prop_1, dict) and isinstance(samp_size, dict):
        if two_sides:
            powerr: dict = {}
            for s in prop_0:
                z = (prop_1[s] - prop_0[s]) / math.sqrt(prop_1[s] * (1 - prop_1[s]) / samp_size[s])
                powerr[s] = normal().cdf(z - z_value) + normal().cdf(-z - z_value)
        else:
            powerr: dict = {}
            for s in prop_0:
                z = (prop_1[s] - prop_0[s]) / math.sqrt(prop_1[s] * (1 - prop_1[s]) / samp_size[s])
                powerr[s] = normal().cdf(z - z_value) + normal().cdf(-z - z_value)
    elif (
        isinstance(prop_0, (int, float))
        and isinstance(prop_1, (int, float))
        and isinstance(samp_size, (int, float))
    ):
        if arcsin:
            h = 2 * math.asin(math.sqrt(prop_1)) - 2 * math.asin(math.sqrt(prop_0))
            if two_sides:
                return (
                    1
                    - normal().cdf(z_value - h * math.sqrt(samp_size))
                    + normal().cdf(-z_value - h * math.sqrt(samp_size))
                )
            else:
                return (
                    1
                    - normal().cdf(z_value - h * math.sqrt(samp_size))
                    # + normal().cdf(-z_value - h * math.sqrt(samp_size))
                )
        else:
            if two_sides:
                z = (prop_1 - prop_0) / math.sqrt(prop_1 * (1 - prop_1) / samp_size)
                return normal().cdf(z - z_value) + normal().cdf(-z - z_value)
            else:
                z = (prop_1 - prop_0) / math.sqrt(prop_1 * (1 - prop_1) / samp_size)
                return normal().cdf(z - z_value) + normal().cdf(-z - z_value)
    elif (
        isinstance(prop_0, (np.np.ndarray, pd.Series, list, tuple))
        and isinstance(prop_1, (np.np.ndarray, pd.Series, list, tuple))
        and isinstance(samp_size, (np.np.ndarray, pd.Series, list, tuple))
    ):
        prop_0 = numpy_array(prop_0)
        prop_1 = numpy_array(prop_1)
        samp_size = numpy_array(samp_size)

        if two_sides:
            z = (prop_1 - prop_0) / np.sqrt(prop_1 * (1 - prop_1) / samp_size)
            return normal().cdf(z - z_value) + normal().cdf(-z - z_value)
        else:
            z = (prop_1 - prop_0) / np.sqrt(prop_1 * (1 - prop_1) / samp_size)
            return normal().cdf(z - z_value) + normal().cdf(-z - z_value)


def calculate_power(
    two_sides: bool,
    delta: Union[DictStrNum, Number, Array],
    sigma: Union[DictStrNum, Number, Array],
    samp_size: Union[DictStrNum, Number, Array],
    alpha: float,
):

    if isinstance(delta, dict) and isinstance(sigma, dict) and isinstance(samp_size, dict):
        if two_sides:
            return {
                s: 1
                - normal().cdf(
                    normal().ppf(1 - alpha / 2) - delta[s] / (sigma[s] / math.sqrt(samp_size[s]))
                )
                + normal().cdf(
                    -normal().ppf(1 - alpha / 2) - delta[s] / (sigma[s] / math.sqrt(samp_size[s]))
                )
                for s in delta
            }
        else:
            return 1 - normal().cdf(
                normal().ppf(1 - alpha) - delta / (sigma / math.sqrt(samp_size))
            )
    elif (
        isinstance(delta, (int, float))
        and isinstance(sigma, (int, float))
        and isinstance(samp_size, (int, float))
    ):
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
    elif (
        isinstance(delta, (np.np.ndarray, pd.Series, list, tuple))
        and isinstance(sigma, (np.np.ndarray, pd.Series, list, tuple))
        and isinstance(samp_size, (np.np.ndarray, pd.Series, list, tuple))
    ):
        delta = numpy_array(delta)
        sigma = numpy_array(sigma)
        power = np.zeros(delta.shape[0])
        for k in range(delta.shape[0]):
            if two_sides:
                power[k] = (
                    1
                    - normal().cdf(
                        normal().ppf(1 - alpha / 2)
                        - delta[k] / (sigma[k] / math.sqrt(samp_size[k]))
                    )
                    + normal().cdf(
                        -normal().ppf(1 - alpha / 2)
                        - delta[k] / (sigma[k] / math.sqrt(samp_size[k]))
                    )
                )
            else:
                power[k] = 1 - normal().cdf(
                    normal().ppf(1 - alpha) - delta[k] / (sigma[k] / math.sqrt(samp_size[k]))
                )
            return power


def sample_size_for_proportion_wald(
    target: Union[DictStrNum, Number, Array],
    half_ci: Union[DictStrNum, Number, Array],
    pop_size: Optional[Union[DictStrNum, Number, Array]],
    deff_c: Union[DictStrNum, Number, Array],
    alpha: float,
) -> Union[DictStrNum, Number, Array]:

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
        isinstance(target, (np.ndarray, pd.Series, list, tuple))
        and isinstance(half_ci, (np.ndarray, pd.Series, list, tuple))
        and isinstance(deff_c, (np.ndarray, pd.Series, list, tuple))
    ):
        target = numpy_array(target)
        half_ci = numpy_array(half_ci)
        deff_c = numpy_array(deff_c)
        samp_size = np.ceil(deff_c * (z_value ** 2) * target * (1 - target) / (half_ci ** 2))
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


def sample_size_for_proportion_fleiss(
    target: Union[DictStrNum, Number, Array],
    half_ci: Union[DictStrNum, Number, Array],
    deff_c: Union[DictStrNum, Number, Array],
    alpha: float,
) -> Union[DictStrNum, Number, Array]:

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
        isinstance(target, (np.ndarray, pd.Series, list, tuple))
        and isinstance(half_ci, (np.ndarray, pd.Series, list, tuple))
        and isinstance(deff_c, (np.ndarray, pd.Series, list, tuple))
    ):
        target = numpy_array(target)
        half_ci = numpy_array(half_ci)
        deff_c = numpy_array(deff_c)
        samp_size = np.zeros(target.shape[0])
        for k in range(target.shape[0]):
            fct = fleiss_factor(target[k], half_ci[k])
            samp_size[k] = math.ceil(
                deff_c[k]
                * (
                    fct * (z_value ** 2) / (4 * half_ci[k] ** 2)
                    + 1 / half_ci[k]
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


def sample_size_for_mean_wald(
    half_ci: Union[DictStrNum, Number, Array],
    sigma: Union[DictStrNum, Number, Array],
    pop_size: Optional[Union[DictStrNum, Number, Array]],
    deff_c: Union[DictStrNum, Number, Array],
    alpha: float,
) -> Union[DictStrNum, Number, Array]:

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
        isinstance(half_ci, (np.ndarray, pd.Series, list, tuple))
        and isinstance(sigma, (np.ndarray, pd.Series, list, tuple))
        and isinstance(deff_c, (np.ndarray, pd.Series, list, tuple))
    ):
        half_ci = numpy_array(half_ci)
        sigma = numpy_array(sigma)
        deff_c = numpy_array(deff_c)
        return np.ceil(deff_c * (z_value * sigma / half_ci) ** 2)
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
            samp_size = sample_size_for_proportion_wald(
                half_ci=self.half_ci,
                target=self.target,
                pop_size=self.pop_size,
                deff_c=self.deff_c,
                alpha=self.alpha,
            )
        elif self.parameter == "proportion" and self.method == "fleiss":
            samp_size = sample_size_for_proportion_fleiss(
                half_ci=self.half_ci,
                target=self.target,
                deff_c=self.deff_c,
                alpha=self.alpha,
            )
        elif self.parameter in ("mean", "total") and self.method == "wald":
            samp_size = sample_size_for_mean_wald(
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


class _SampleSizeForDifference:
    """Internal class to compute sample size of difference"""

    def __init__(
        self,
        parameter: str = "proportion",
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = False,
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
    def _calculate_ss_wald(
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

        self._input_parameters_validation(
            delta=delta, sigma=sigma, deff=deff, resp_rate=resp_rate, number_strata=number_strata
        )

        self.alpha = alpha
        self.power = power

        # samp_size: Union[DictStrNum, Number]
        if self.parameter in ("proportion", "mean", "total") and self.method == "wald":
            self.samp_size = self._calculate_ss_wald(
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


class SampleSizeOneMean(_SampleSizeForDifference):
    """SampleSizeOneProportion implements sample size calculation for one mean"""

    def __init__(
        self,
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        _SampleSizeForDifference.__init__(
            self,
            parameter="mean",
            method=method,
            stratification=stratification,
            two_sides=two_sides,
            params_estimated=params_estimated,
        )

        self.mean_0: Union[DictStrNum, Number]
        self.mean_1: Union[DictStrNum, Number]

    def calculate(
        self,
        mean_0: Union[DictStrNum, Number],
        mean_1: Union[DictStrNum, Number],
        sigma: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> None:

        delta: Union[DictStrNum, Number]
        if isinstance(mean_0, (int, float)) and isinstance(mean_1, (int, float)):
            delta = mean_1 - mean_0
        elif isinstance(mean_0, dict) and isinstance(mean_1, dict):
            delta = {k: mean_1[k] - mean_0[k] for k in mean_0}
        else:
            raise AssertionError("target_0 and targget_1 are not the same type.")

        self._input_parameters_validation(
            delta=delta, sigma=sigma, deff=deff, resp_rate=resp_rate, number_strata=number_strata
        )

        self.alpha = alpha
        self.power = power

        # samp_size: Union[DictStrNum, Number]
        self.samp_size = self._calculate_ss_wald(
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


class SampleSizeOneTotal(_SampleSizeForDifference):
    """SampleSizeOneProportion implements sample size calculation for one mean"""

    def __init__(
        self,
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        _SampleSizeForDifference.__init__(
            self,
            parameter="total",
            method=method,
            stratification=stratification,
            two_sides=two_sides,
            params_estimated=params_estimated,
        )

        self.total_0: Union[DictStrNum, Number]
        self.total_1: Union[DictStrNum, Number]

    def calculate(
        self,
        total_0: Union[DictStrNum, Number],
        total_1: Union[DictStrNum, Number],
        sigma: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        # pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> None:

        delta: Union[DictStrNum, Number]
        if isinstance(total_0, (int, float)) and isinstance(total_1, (int, float)):
            delta = total_1 - total_0
        elif isinstance(total_0, dict) and isinstance(total_1, dict):
            delta = {k: total_1[k] - total_0[k] for k in total_0}
        else:
            raise AssertionError("target_0 and targget_1 are not the same type.")

        self._input_parameters_validation(
            delta=delta, sigma=sigma, deff=deff, resp_rate=resp_rate, number_strata=number_strata
        )

        self.alpha = alpha
        self.power = power

        # samp_size: Union[DictStrNum, Number]
        self.samp_size = self._calculate_ss_wald(
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


class SampleSizeOneProportion(_SampleSizeForDifference):
    """SampleSizeOneProportion implements sample size calculation for one mean"""

    def __init__(
        self,
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
    ) -> None:

        _SampleSizeForDifference.__init__(
            self,
            parameter="proportion",
            method=method,
            stratification=stratification,
            two_sides=two_sides,
        )

        self.proportion_0: Union[DictStrNum, Number]
        self.proportion_1: Union[DictStrNum, Number]

        del self.params_estimated

    def _input_parameters_validation(
        self,
        prop_0: Union[DictStrNum, Number],
        prop_1: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
    ) -> None:

        is_prop0_dict = isinstance(prop_0, dict)
        is_prop1_dict = isinstance(prop_1, dict)
        is_deff_dict = isinstance(deff, dict)
        is_resp_rate_dict = isinstance(resp_rate, dict)

        number_dictionaries = (
            is_prop0_dict + is_prop1_dict + is_deff_dict + is_resp_rate_dict  # + is_pop_size_dict
        )

        if self.parameter == "proportion":
            if (isinstance(prop_0, (int, float)) and not 0 <= prop_0 <= 1) or (
                isinstance(prop_1, (int, float)) and not 0 <= prop_1 <= 1
            ):
                raise ValueError("Target for proportions must be between 0 and 1.")
            if isinstance(prop_0, dict) and isinstance(prop_1, dict):
                for s in prop_0:
                    if (not 0 <= prop_0[s] <= 1) or (not 0 <= prop_1[s] <= 1):
                        raise ValueError("Target for proportions must be between 0 and 1.")

        stratum: Optional[list[StringNumber]] = None
        if not self.stratification and (
            isinstance(prop_0, dict)
            or isinstance(prop_1, dict)
            or isinstance(deff, dict)
            or isinstance(resp_rate, dict)
            # or isinstance(pop_size, dict)
        ):
            raise AssertionError("No python dictionary needed for non-stratified sample.")
        elif (
            not self.stratification
            and isinstance(prop_0, (int, float))
            and isinstance(prop_1, (int, float))
            and isinstance(deff, (int, float))
            and isinstance(resp_rate, (int, float))
        ):
            self.prop_0 = prop_0
            self.prop_1 = prop_1
            self.deff_c = deff
            self.resp_rate = resp_rate
        elif (
            self.stratification
            and isinstance(prop_0, (int, float))
            and isinstance(prop_1, (int, float))
            and isinstance(deff, (int, float))
            and isinstance(resp_rate, (int, float))
        ):
            if number_strata is not None:
                stratum = ["_stratum_" + str(i) for i in range(1, number_strata + 1)]
                self.target = dict(zip(stratum, np.repeat(prop_0, number_strata)))
                self.sigma = dict(zip(stratum, np.repeat(prop_1, number_strata)))
                self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
                self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
                # if isinstance(pop_size, (int, float)):
                #     self.pop_size = dict(zip(stratum, np.repeat(pop_size, number_strata)))
            else:
                raise ValueError("Number of strata not specified!")
        elif self.stratification and number_dictionaries > 0:
            dict_number = 0
            for ll in [prop_0, prop_1, deff, resp_rate]:  # , pop_size]:
                if isinstance(ll, dict):
                    dict_number += 1
                    if dict_number == 1:
                        stratum = list(ll.keys())
                    elif dict_number > 0:
                        if stratum != list(ll.keys()):
                            raise AssertionError("Python dictionaries have different keys")
            number_strata = len(stratum) if stratum is not None else 0
            if not is_prop0_dict and isinstance(prop_0, (int, float)) and stratum is not None:
                self.prop_0 = dict(zip(stratum, np.repeat(prop_0, number_strata)))
            elif isinstance(prop_0, dict):
                self.prop_0 = prop_0
            if not is_prop1_dict and isinstance(prop_1, (int, float)) and stratum is not None:
                self.prop1 = dict(zip(stratum, np.repeat(prop_1, number_strata)))
            elif isinstance(prop_1, dict):
                self.prop_1 = prop_1
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
    def _calculate_ss_wald(
        delta: Union[DictStrNum, Number],
        prop_0: Union[DictStrNum, Number],
        prop_1: Union[DictStrNum, Number],
        arcsin: bool,
        deff_c: Union[DictStrNum, Number],
        z_alpha: float,
        z_beta: float,
    ) -> Union[DictStrNum, Number]:

        if isinstance(prop_0, dict) and isinstance(prop_1, dict) and isinstance(deff_c, dict):
            samp_size: DictStrNum = {}
            for s in prop_0:
                if arcsin:
                    samp_size[s] = math.ceil(deff_c[s] * ((z_alpha + z_beta) / delta[s]) ** 2)
                else:
                    samp_size[s] = math.ceil(
                        deff_c[s]
                        * (
                            (
                                z_alpha * math.sqrt(prop_0[s] * (1 - prop_0[s]))
                                + z_beta * math.sqrt(prop_1[s] * (1 - prop_1[s]))
                            )
                            / delta[s]
                        )
                        ** 2
                    )
            return samp_size
        elif (
            isinstance(prop_0, (int, float))
            and isinstance(prop_1, (int, float))
            and isinstance(deff_c, (int, float))
        ):
            if arcsin:
                return math.ceil(deff_c * ((z_alpha + z_beta) / delta) ** 2)
            else:
                return math.ceil(
                    deff_c
                    * (
                        (
                            z_alpha * math.sqrt(prop_0 * (1 - prop_0))
                            + z_beta * math.sqrt(prop_1 * (1 - prop_1))
                        )
                        / delta
                    )
                    ** 2
                )
        else:
            raise TypeError("target, half_ci, and sigma must be numbers or dictionaries!")

    def calculate(
        self,
        prop_0: Union[DictStrNum, Number],
        prop_1: Union[DictStrNum, Number],
        arcsin: bool = False,
        continuity: bool = False,
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        # pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> None:

        const = 1 if self.two_sides else 2
        self.arcsin = arcsin

        self._input_parameters_validation(
            prop_0=prop_0,
            prop_1=prop_1,
            deff=deff,
            resp_rate=resp_rate,
            number_strata=number_strata,
        )

        delta: Union[DictStrNum, Number]
        if isinstance(prop_0, (int, float)) and isinstance(prop_1, (int, float)):
            if self.arcsin:
                delta = const * math.asin(math.sqrt(prop_0)) - const * math.asin(math.sqrt(prop_1))
            else:
                delta = prop_1 - prop_0
            sigma = prop_1 * (1 - prop_1)

        elif isinstance(prop_0, dict) and isinstance(prop_1, dict):
            if self.arcsin:
                delta = {
                    k: const * math.asin(math.sqrt(prop_0[k]))
                    - const * math.asin(math.sqrt(prop_1[k]))
                    for k in prop_0
                }
            else:
                delta = {k: prop_1[k] - prop_0[k] for k in prop_0}
            sigma = {k: prop_1[k] * (1 - prop_0[k]) for k in prop_1}
        else:
            raise AssertionError("target_0 and target_1 are not the same type.")

        self.delta = delta
        self.sigma = sigma
        self.alpha = alpha
        self.power = power

        # self.samp_size = samp_size
        if self.two_sides:
            z_alpha = normal().ppf(1 - self.alpha / 2)
        else:
            z_alpha = normal().ppf(1 - self.alpha)
        z_beta = normal().ppf(self.power)

        # samp_size: Union[DictStrNum, Number]
        self.samp_size = self._calculate_ss_wald(
            delta=self.delta,
            prop_0=prop_0,
            prop_1=prop_1,
            arcsin=self.arcsin,
            deff_c=self.deff_c,
            z_alpha=z_alpha,
            z_beta=z_beta,
        )

        if self.stratification:
            if continuity:
                for s in self.samp_size:
                    self.samp_size[s] = math.ceil(
                        self.samp_size[s]
                        + 1
                        / (
                            z_alpha * math.sqrt(prop_0[s] * (1 - prop_0[s]) / self.samp_size[s])
                            + z_beta * math.sqrt(prop_1[s] * (1 - prop_1[s]))
                        )
                    )

            self.actual_power[s] = power_for_proportion(
                self.two_sides, self.delta[s], self.sigma[s], self.samp_size[s], self.alpha
            )

        else:
            if continuity:
                self.samp_size = math.ceil(
                    self.samp_size
                    + 1
                    / (
                        (
                            z_alpha * math.sqrt(prop_0 * (1 - prop_0) / self.samp_size)
                            + z_beta * math.sqrt(prop_1 * (1 - prop_1))
                        )
                    )
                )

            self.actual_power = power_for_proportion(
                self.two_sides, self.delta, self.sigma, self.samp_size, self.alpha
            )
