"""Sample size calculation module 

"""

from __future__ import annotations

import math

from typing import Optional, Union

import numpy as np
import pandas as pd

from scipy.stats import norm as normal
from scipy.stats import t as student
from samplics.sampling.size_functions import (
    calculate_ss_wald_mean_one_sample,
    calculate_ss_wald_mean_two_sample,
    calculate_ss_wald_prop_two_sample,
)

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
        isinstance(prop_0, (np.ndarray, pd.Series, list, tuple))
        and isinstance(prop_1, (np.ndarray, pd.Series, list, tuple))
        and isinstance(samp_size, (np.ndarray, pd.Series, list, tuple))
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
        isinstance(delta, (np.ndarray, pd.Series, list, tuple))
        and isinstance(sigma, (np.ndarray, pd.Series, list, tuple))
        and isinstance(samp_size, (np.ndarray, pd.Series, list, tuple))
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


def _ss_for_proportion_wald(
    target: Union[Number, Array],
    half_ci: Union[Number, Array],
    pop_size: Optional[Union[Number, Array]],
    deff_c: Union[Number, Array],
    resp_rate: Union[Number, Array],
    alpha: float,
) -> Union[Number, Array]:

    if isinstance(target, (np.ndarray, pd.Series, list, tuple)):
        target = numpy_array(target)
    if isinstance(half_ci, (np.ndarray, pd.Series, list, tuple)):
        half_ci = numpy_array(half_ci)
    if isinstance(deff_c, (np.ndarray, pd.Series, list, tuple)):
        deff_c = numpy_array(deff_c)
    if isinstance(resp_rate, (np.ndarray, pd.Series, list, tuple)):
        resp_rate = numpy_array(resp_rate)
    if isinstance(pop_size, (np.ndarray, pd.Series, list, tuple)):
        pop_size = numpy_array(pop_size)
    if isinstance(alpha, (np.ndarray, pd.Series, list, tuple)):
        alpha = numpy_array(alpha)

    z_value = normal().ppf(1 - alpha / 2)

    if isinstance(pop_size, (np.ndarray, int, float)):
        return math.ceil(
            (1 / resp_rate)
            * deff_c
            * pop_size
            * z_value**2
            * target
            * (1 - target)
            / ((pop_size - 1) * half_ci**2 + z_value**2 * target * (1 - target))
        )
    else:
        return math.ceil(
            (1 / resp_rate) * deff_c * z_value**2 * target * (1 - target) / half_ci**2
        )


def _ss_for_proportion_wald_stratified(
    target: DictStrNum,
    half_ci: DictStrNum,
    pop_size: Optional[DictStrNum],
    deff_c: DictStrNum,
    resp_rate: DictStrNum,
    alpha: DictStrNum,
) -> DictStrNum:

    samp_size: DictStrNum = {}
    for s in half_ci:
        pop_size_c = None if pop_size is None else pop_size[s]
        samp_size[s] = _ss_for_proportion_wald(
            target=target[s],
            half_ci=half_ci[s],
            pop_size=pop_size_c,
            deff_c=deff_c[s],
            resp_rate=resp_rate[s],
            alpha=alpha[s],
        )

    return samp_size


def sample_size_for_proportion_wald(
    target: Union[DictStrNum, Number, Array],
    half_ci: Union[DictStrNum, Number, Array],
    pop_size: Optional[Union[DictStrNum, Number, Array]] = None,
    deff_c: Union[DictStrNum, Number, Array] = 1.0,
    resp_rate: Union[DictStrNum, Number, Array] = 1.0,
    alpha: Union[DictStrNum, Number, Array] = 0.05,
    stratification: bool = False,
) -> Union[DictStrNum, Number, Array]:

    if stratification:
        return _ss_for_proportion_wald_stratified(
            target=target,
            half_ci=half_ci,
            pop_size=pop_size,
            deff_c=deff_c,
            resp_rate=resp_rate,
            alpha=alpha,
        )
    else:
        return _ss_for_proportion_wald(
            target=target,
            half_ci=half_ci,
            pop_size=pop_size,
            deff_c=deff_c,
            resp_rate=resp_rate,
            alpha=alpha,
        )


def _ss_for_proportion_fleiss(
    target: Union[Number, Array],
    half_ci: Union[Number, Array],
    deff_c: Union[Number, Array],
    resp_rate: Union[Number, Array],
    alpha: Union[Number, Array],
) -> Union[Number, Array]:

    if isinstance(target, (np.ndarray, pd.Series, list, tuple)):
        target = numpy_array(target)
    if isinstance(half_ci, (np.ndarray, pd.Series, list, tuple)):
        half_ci = numpy_array(half_ci)
    if isinstance(deff_c, (np.ndarray, pd.Series, list, tuple)):
        deff_c = numpy_array(deff_c)
    if isinstance(resp_rate, (np.ndarray, pd.Series, list, tuple)):
        resp_rate = numpy_array(resp_rate)
    if isinstance(alpha, (np.ndarray, pd.Series, list, tuple)):
        alpha = numpy_array(alpha)

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

    if isinstance(target, np.ndarray):
        fct = np.zeros(target.shape[0])
        for k in range(target.shape[0]):
            fct[k] = fleiss_factor(target[k], half_ci[k])
    else:
        fct = fleiss_factor(target, half_ci)

    return math.ceil(
        (1 / resp_rate)
        * deff_c
        * (
            fct * (z_value**2) / (4 * half_ci**2)
            + 1 / half_ci
            - 2 * z_value**2
            + (z_value + 2) / fct
        )
    )


def _ss_for_proportion_fleiss_stratified(
    target: DictStrNum,
    half_ci: DictStrNum,
    deff_c: DictStrNum,
    resp_rate: DictStrNum,
    alpha: DictStrNum,
) -> DictStrNum:

    samp_size: DictStrNum = {}
    for s in half_ci:
        samp_size[s] = _ss_for_proportion_fleiss(
            target=target[s],
            half_ci=half_ci[s],
            deff_c=deff_c[s],
            resp_rate=resp_rate[s],
            alpha=alpha[s],
        )

    return samp_size


def sample_size_for_proportion_fleiss(
    target: Union[DictStrNum, Number, Array],
    half_ci: Union[DictStrNum, Number, Array],
    deff_c: Union[DictStrNum, Number, Array] = 1.0,
    resp_rate: Union[DictStrNum, Number, Array] = 1.0,
    alpha: Union[DictStrNum, Number, Array] = 0.05,
    stratification: bool = False,
) -> Union[DictStrNum, Number, Array]:

    if stratification:
        return _ss_for_proportion_fleiss_stratified(
            target=target, half_ci=half_ci, deff_c=deff_c, resp_rate=resp_rate, alpha=alpha
        )
    else:
        return _ss_for_proportion_fleiss(
            target=target, half_ci=half_ci, deff_c=deff_c, resp_rate=resp_rate, alpha=alpha
        )


def _ss_for_mean_wald(
    half_ci: Union[Number, Array],
    sigma: Union[Number, Array],
    pop_size: Optional[Union[Number, Array]],
    deff_c: Union[Number, Array],
    resp_rate: Union[Number, Array],
    alpha: Union[Number, Array],
) -> Union[Number, Array]:

    if isinstance(half_ci, (np.ndarray, pd.Series, list, tuple)):
        half_ci = numpy_array(half_ci)
    if isinstance(sigma, (np.ndarray, pd.Series, list, tuple)):
        sigma = numpy_array(sigma)
    if isinstance(deff_c, (np.ndarray, pd.Series, list, tuple)):
        deff_c = numpy_array(deff_c)
    if isinstance(resp_rate, (np.ndarray, pd.Series, list, tuple)):
        resp_rate = numpy_array(resp_rate)
    if isinstance(pop_size, (np.ndarray, pd.Series, list, tuple)):
        pop_size = numpy_array(pop_size)
    if isinstance(alpha, (np.ndarray, pd.Series, list, tuple)):
        alpha = numpy_array(alpha)

    z_value = normal().ppf(1 - alpha / 2)
    if isinstance(pop_size, (np.ndarray, int, float)):
        return math.ceil(
            ((1 / resp_rate) * deff_c * pop_size * z_value**2 * sigma**2)
            / ((pop_size - 1) * half_ci**2 + z_value**2 * sigma**2)
        )
    else:
        return math.ceil((1 / resp_rate) * deff_c * z_value**2 * sigma**2 / half_ci**2)


def _ss_for_mean_wald_stratified(
    half_ci: DictStrNum,
    sigma: Union[Number, Array],
    pop_size: Optional[DictStrNum],
    deff_c: DictStrNum,
    resp_rate: DictStrNum,
    alpha: DictStrNum,
) -> DictStrNum:

    samp_size: DictStrNum = {}
    for s in half_ci:
        pop_size_c = None if pop_size is None else pop_size[s]
        samp_size[s] = _ss_for_mean_wald(
            half_ci=half_ci[s],
            sigma=sigma[s],
            pop_size=pop_size_c,
            deff_c=deff_c[s],
            resp_rate=resp_rate[s],
            alpha=alpha[s],
        )

    return samp_size


def sample_size_for_mean_wald(
    half_ci: Union[DictStrNum, Number, Array],
    sigma: Union[DictStrNum, Number, Array],
    pop_size: Optional[Union[DictStrNum, Number, Array]] = None,
    deff_c: Union[DictStrNum, Number, Array] = 1.0,
    resp_rate: Union[DictStrNum, Number] = 1.0,
    alpha: Union[DictStrNum, Number, Array] = 0.05,
    stratification: bool = False,
) -> Union[DictStrNum, Number, Array]:

    if stratification:
        return _ss_for_mean_wald_stratified(
            half_ci=half_ci,
            sigma=sigma,
            pop_size=pop_size,
            deff_c=deff_c,
            resp_rate=resp_rate,
            alpha=alpha,
        )
    else:
        return _ss_for_mean_wald(
            half_ci=half_ci,
            sigma=sigma,
            pop_size=pop_size,
            deff_c=deff_c,
            resp_rate=resp_rate,
            alpha=alpha,
        )


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
        self.half_ci: Union[DictStrNum, Number]
        self.samp_size: Union[DictStrNum, Number] = 0
        self.deff_c: Union[DictStrNum, Number] = 1.0
        self.deff_w: Union[DictStrNum, Number] = 1.0
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
        half_ci: Union[DictStrNum, Number],
        target: Optional[Union[DictStrNum, Number]] = None,
        sigma: Optional[Union[DictStrNum, Number]] = None,
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Number]] = None,
        alpha: float = 0.05,
    ) -> None:

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

        if self.stratification:
            (
                self.half_ci,
                self.target,
                self.sigma,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
            ) = convert_numbers_to_dicts(
                number_strata, half_ci, target, sigma, deff, resp_rate, pop_size, alpha
            )
        else:
            (
                self.half_ci,
                self.target,
                self.sigma,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
            ) = (half_ci, target, sigma, deff, resp_rate, pop_size, alpha)

        samp_size: Union[DictStrNum, Number]
        if self.parameter == "proportion" and self.method == "wald":
            self.samp_size = sample_size_for_proportion_wald(
                half_ci=self.half_ci,
                target=self.target,
                pop_size=self.pop_size,
                deff_c=self.deff_c,
                resp_rate=self.resp_rate,
                alpha=self.alpha,
                stratification=self.stratification,
            )
        elif self.parameter == "proportion" and self.method == "fleiss":
            self.samp_size = sample_size_for_proportion_fleiss(
                half_ci=self.half_ci,
                target=self.target,
                deff_c=self.deff_c,
                resp_rate=self.resp_rate,
                alpha=self.alpha,
                stratification=self.stratification,
            )
        elif self.parameter in ("mean", "total") and self.method == "wald":
            self.samp_size = sample_size_for_mean_wald(
                half_ci=self.half_ci,
                sigma=self.sigma,
                pop_size=self.pop_size,
                deff_c=self.deff_c,
                resp_rate=self.resp_rate,
                alpha=self.alpha,
                stratification=self.stratification,
            )

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


class SampleSizeMeanOneSample:
    """SampleSizeMeanOneSample implements sample size calculation for mean under one-sample design"""

    def __init__(
        self,
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        self.parameter = "mean"
        self.method = method.lower()
        if self.method not in ("wald"):
            raise AssertionError("The method must be wald.")

        self.stratification = stratification
        self.two_sides = two_sides
        self.params_estimated = params_estimated

        self.samp_size: Union[DictStrNum, Array, Number]
        self.actual_power: Union[DictStrNum, Array, Number]

        self.mean_0: Union[DictStrNum, Array, Number]
        self.mean_1: Union[DictStrNum, Array, Number]
        self.epsilon: Union[DictStrNum, Array, Number]
        self.delta: Union[DictStrNum, Array, Number]
        self.sigma: Union[DictStrNum, Array, Number]

        self.deff_c: Union[DictStrNum, Array, Number]
        self.deff_w: Union[DictStrNum, Array, Number]
        self.resp_rate: Union[DictStrNum, Array, Number]
        self.pop_size: Optional[Union[DictStrNum, Array, Number]] = None

        self.alpha: Union[DictStrNum, Array, Number]
        self.beta: Union[DictStrNum, Array, Number]
        self.power: Union[DictStrNum, Array, Number]

    def calculate(
        self,
        mean_0: Union[DictStrNum, Array, Number],
        mean_1: Union[DictStrNum, Array, Number],
        sigma: Union[DictStrNum, Array, Number],
        delta: Union[DictStrNum, Array, Number] = 0.0,
        deff: Union[DictStrNum, Array, Number] = 1.0,
        resp_rate: Union[DictStrNum, Array, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Array, Number]] = None,
        alpha: Union[DictStrNum, Array, Number] = 0.05,
        power: Union[DictStrNum, Array, Number] = 0.80,
    ) -> None:

        if self.stratification:
            (
                self.mean_0,
                self.mean_1,
                self.sigma,
                self.delta,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
                self.power,
            ) = convert_numbers_to_dicts(
                number_strata,
                mean_0,
                mean_1,
                sigma,
                delta,
                deff,
                resp_rate,
                pop_size,
                alpha,
                power,
            )
        else:
            (
                self.mean_0,
                self.mean_1,
                self.sigma,
                self.delta,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
                self.power,
            ) = (
                mean_0,
                mean_1,
                sigma,
                delta,
                deff,
                resp_rate,
                pop_size,
                alpha,
                power,
            )

        if self.stratification:
            epsilon: DictStrNum = {}
            for s in mean_0:
                epsilon[s] = mean_1[s] - mean_0[s]
            self.epsilon = epsilon
        else:
            self.epsilon = mean_1 - mean_0

        self.samp_size = calculate_ss_wald_mean_one_sample(
            two_sides=self.two_sides,
            epsilon=self.epsilon,
            delta=self.delta,
            sigma=self.sigma,
            deff_c=self.deff_c,
            resp_rate=self.resp_rate,
            alpha=self.alpha,
            power=self.power,
            stratification=self.stratification,
        )

        if self.stratification:
            actual_power: DictStrNum = {}
            for k in self.samp_size:
                actual_power[k] = calculate_power(
                    self.two_sides,
                    self.epsilon[k],
                    self.sigma[k],
                    self.samp_size[k],
                    self.alpha[k],
                )
            self.actual_power = actual_power
        else:
            self.actual_power = calculate_power(
                self.two_sides, self.epsilon, self.sigma, self.samp_size, self.alpha
            )


class SampleSizePropOneSample:
    """SampleSizePropOneSample implements sample size calculation for propoertion under one-sample design"""

    def __init__(
        self,
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        self.parameter = "proportion"
        self.method = method.lower()
        if self.method not in ("wald"):
            raise AssertionError("The method must be wald.")

        self.stratification = stratification
        self.two_sides = two_sides
        self.params_estimated = params_estimated

        self.samp_size: Union[DictStrNum, Number]
        self.actual_power: Union[DictStrNum, Number]

        self.prop_0: Union[DictStrNum, Number]
        self.prop_1: Union[DictStrNum, Number]
        self.epsilon: Union[DictStrNum, Number]
        self.delta: Union[DictStrNum, Number]
        self.sigma: Union[DictStrNum, Number]

        self.deff_c: Union[DictStrNum, Number]
        self.deff_w: Union[DictStrNum, Number]
        self.resp_rate: Union[DictStrNum, Number]
        self.pop_size: Optional[Union[DictStrNum, Number]] = None

        self.alpha: Union[DictStrNum, Number]
        self.beta: Union[DictStrNum, Number]
        self.power: Union[DictStrNum, Number]

    def calculate(
        self,
        prop_0: Union[DictStrNum, Array, Number],
        prop_1: Union[DictStrNum, Array, Number],
        delta: Union[DictStrNum, Array, Number] = 0.0,
        arcsin: bool = False,
        continuity: bool = False,
        deff: Union[DictStrNum, Array, Number] = 1.0,
        resp_rate: Union[DictStrNum, Array, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Array, Number]] = None,
        alpha: Union[DictStrNum, Array, Number] = 0.05,
        power: Union[DictStrNum, Array, Number] = 0.80,
    ) -> None:

        if self.stratification:
            (
                self.prop_0,
                self.prop_1,
                self.delta,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
                self.power,
            ) = convert_numbers_to_dicts(
                number_strata,
                prop_0,
                prop_1,
                delta,
                deff,
                resp_rate,
                pop_size,
                alpha,
                power,
            )
        else:
            (
                self.prop_0,
                self.prop_1,
                self.delta,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
                self.power,
            ) = (
                prop_0,
                prop_1,
                delta,
                deff,
                resp_rate,
                pop_size,
                alpha,
                power,
            )

        if self.stratification:
            epsilon: DictStrNum = {}
            sigma: DictStrNum = {}
            for s in prop_0:
                epsilon[s] = prop_1[s] - prop_0[s]
                sigma[s] = math.sqrt(prop_1[s] * (1 - prop_1[s]))
            self.epsilon = epsilon
            self.sigma = sigma
        else:
            self.epsilon = prop_1 - prop_0
            self.sigma = math.sqrt(prop_1 * (1 - prop_1))

        self.arcsin = arcsin
        self.continuity = continuity

        self.samp_size = calculate_ss_wald_mean_one_sample(
            two_sides=self.two_sides,
            epsilon=self.epsilon,
            delta=self.delta,
            sigma=self.sigma,
            deff_c=self.deff_c,
            resp_rate=self.resp_rate,
            alpha=self.alpha,
            power=self.power,
            stratification=self.stratification,
        )

        if self.stratification:
            actual_power: DictStrNum = {}
            for k in self.samp_size:
                actual_power[k] = calculate_power(
                    self.two_sides,
                    self.epsilon[k],
                    self.sigma[k],
                    self.samp_size[k],
                    self.alpha[k],
                )
            self.actual_power = actual_power
        else:
            self.actual_power = calculate_power(
                self.two_sides, self.epsilon, self.sigma, self.samp_size, self.alpha
            )


class SampleSizeMeanTwoSample:
    """SampleSizeMeanTwoSample implements sample size calculation for mean under two-sample design"""

    def __init__(
        self,
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        self.parameter = "mean"
        self.method = method.lower()
        if self.method not in ("wald"):
            raise AssertionError("The method must be wald.")

        self.stratification = stratification
        self.two_sides = two_sides
        self.params_estimated = params_estimated

        self.samp_size: Union[DictStrNum, Array, Number]
        self.actual_power: Union[DictStrNum, Array, Number]

        self.mean_1: Union[DictStrNum, Array, Number]
        self.mean_2: Union[DictStrNum, Array, Number]
        self.epsilon: Union[DictStrNum, Array, Number]
        self.delta: Union[DictStrNum, Array, Number]
        self.sigma_1: Union[DictStrNum, Array, Number]
        self.sigma_2: Union[DictStrNum, Array, Number]
        self.equal_variance: Union[DictStrNum, Array, Number]

        self.deff_c: Union[DictStrNum, Array, Number]
        self.deff_w: Union[DictStrNum, Array, Number]
        self.resp_rate: Union[DictStrNum, Array, Number]
        self.pop_size: Optional[Union[DictStrNum, Array, Number]] = None

        self.alpha: Union[DictStrNum, Array, Number]
        self.beta: Union[DictStrNum, Array, Number]
        self.power: Union[DictStrNum, Array, Number]

    def calculate(
        self,
        mean_1: Union[DictStrNum, Array, Number],
        mean_2: Union[DictStrNum, Array, Number],
        sigma_1: Union[DictStrNum, Array, Number],
        sigma_2: Optional[Union[DictStrNum, Array, Number]] = None,
        equal_variance: Union[DictStrNum, Array, Number] = True,
        kappa: Optional[Union[DictStrNum, Array, Number]] = 1,
        delta: Union[DictStrNum, Array, Number] = 0.0,
        deff: Union[DictStrNum, Array, Number] = 1.0,
        resp_rate: Union[DictStrNum, Array, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Array, Number]] = None,
        alpha: Union[DictStrNum, Array, Number] = 0.05,
        power: Union[DictStrNum, Array, Number] = 0.80,
    ) -> None:

        if self.stratification:
            (
                self.mean_1,
                self.mean_2,
                self.sigma_1,
                self.sigma_2,
                self.equal_variance,
                self.kappa,
                self.delta,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
                self.power,
            ) = convert_numbers_to_dicts(
                number_strata,
                mean_1,
                mean_2,
                sigma_1,
                sigma_2,
                equal_variance,
                kappa,
                delta,
                deff,
                resp_rate,
                pop_size,
                alpha,
                power,
            )
        else:
            (
                self.mean_1,
                self.mean_2,
                self.sigma_1,
                self.sigma_2,
                self.equal_variance,
                self.kappa,
                self.delta,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
                self.power,
            ) = (
                mean_1,
                mean_2,
                sigma_1,
                sigma_2,
                equal_variance,
                kappa,
                delta,
                deff,
                resp_rate,
                pop_size,
                alpha,
                power,
            )

        if self.stratification:
            epsilon: DictStrNum = {}
            for s in mean_1:
                epsilon[s] = mean_2[s] - mean_1[s]
            self.epsilon = epsilon
        else:
            self.epsilon = mean_2 - mean_1

        self.samp_size = calculate_ss_wald_mean_two_sample(
            two_sides=self.two_sides,
            epsilon=self.epsilon,
            delta=self.delta,
            sigma_1=self.sigma_1,
            sigma_2=self.sigma_2,
            equal_variance=self.equal_variance,
            kappa=kappa,
            deff_c=self.deff_c,
            resp_rate=resp_rate,
            alpha=self.alpha,
            power=self.power,
            stratification=self.stratification,
        )

        # if self.stratification:
        #     for k in self.samp_size:
        #         self.actual_power[k] = calculate_power(
        #             self.two_sides,
        #             self.epsilon[k],
        #             self.sigma[k],
        #             self.samp_size[k],
        #             self.alpha[k],
        #         )
        # else:
        #     self.actual_power = calculate_power(
        #         self.two_sides, self.epsilon, self.sigma, self.samp_size, self.alpha
        #     )


class SampleSizePropTwoSample:
    """SampleSizeMeanTwoSample implements sample size calculation for mean under two-sample design"""

    def __init__(
        self,
        method: str = "wald",
        stratification: bool = False,
        two_sides: bool = True,
        params_estimated: bool = True,
    ) -> None:

        self.parameter = "proportion"
        self.method = method.lower()
        if self.method not in ("wald"):
            raise AssertionError("The method must be wald.")

        self.stratification = stratification
        self.two_sides = two_sides
        self.params_estimated = params_estimated

        self.samp_size: Union[DictStrNum, Array, Number]
        self.actual_power: Union[DictStrNum, Array, Number]

        self.prop_1: Union[DictStrNum, Array, Number]
        self.prop_2: Union[DictStrNum, Array, Number]
        self.epsilon: Union[DictStrNum, Array, Number]
        self.delta: Union[DictStrNum, Array, Number]

        self.deff_c: Union[DictStrNum, Array, Number]
        self.deff_w: Union[DictStrNum, Array, Number]
        self.resp_rate: Union[DictStrNum, Array, Number]
        self.pop_size: Optional[Union[DictStrNum, Array, Number]] = None

        self.alpha: Union[DictStrNum, Array, Number]
        self.beta: Union[DictStrNum, Array, Number]
        self.power: Union[DictStrNum, Array, Number]

    def calculate(
        self,
        prop_1: Union[DictStrNum, Array, Number],
        prop_2: Union[DictStrNum, Array, Number],
        kappa: Optional[Union[DictStrNum, Array, Number]] = 1,
        delta: Union[DictStrNum, Array, Number] = 0.0,
        deff: Union[DictStrNum, Array, Number] = 1.0,
        resp_rate: Union[DictStrNum, Array, Number] = 1.0,
        number_strata: Optional[int] = None,
        pop_size: Optional[Union[DictStrNum, Array, Number]] = None,
        alpha: Union[DictStrNum, Array, Number] = 0.05,
        power: Union[DictStrNum, Array, Number] = 0.80,
    ) -> None:

        if self.stratification:
            (
                self.prop_1,
                self.prop_2,
                self.kappa,
                self.delta,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
                self.power,
            ) = convert_numbers_to_dicts(
                number_strata,
                prop_1,
                prop_2,
                kappa,
                delta,
                deff,
                resp_rate,
                pop_size,
                alpha,
                power,
            )
        else:
            (
                self.prop_1,
                self.prop_2,
                self.kappa,
                self.delta,
                self.deff_c,
                self.resp_rate,
                self.pop_size,
                self.alpha,
                self.power,
            ) = (
                prop_1,
                prop_2,
                kappa,
                delta,
                deff,
                resp_rate,
                pop_size,
                alpha,
                power,
            )

        if self.stratification:
            epsilon: DictStrNum = {}
            for s in prop_1:
                epsilon[s] = prop_2[s] - prop_1[s]
            self.epsilon = epsilon
        else:
            self.epsilon = prop_2 - prop_1

        self.samp_size = calculate_ss_wald_prop_two_sample(
            two_sides=self.two_sides,
            epsilon=self.epsilon,
            delta=self.delta,
            prop_1=self.prop_1,
            prop_2=self.prop_2,
            kappa=kappa,
            deff_c=self.deff_c,
            alpha=self.alpha,
            power=self.power,
            stratification=self.stratification,
        )

        # if self.stratification:
        #     for k in self.samp_size:
        #         self.actual_power[k] = calculate_power(
        #             self.two_sides,
        #             self.epsilon[k],
        #             self.sigma[k],
        #             self.samp_size[k],
        #             self.alpha[k],
        #         )
        # else:
        #     self.actual_power = calculate_power(
        #         self.two_sides, self.epsilon, self.sigma, self.samp_size, self.alpha
        #     )
