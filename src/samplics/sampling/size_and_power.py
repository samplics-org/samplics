"""Sample size calculation module 

"""

from __future__ import annotations

import math

from typing import Optional, Union

import numpy as np
import pandas as pd

from scipy.stats import norm as normal

from samplics.utils.checks import assert_proportions
from samplics.utils.formats import numpy_array
from samplics.utils.types import Array, DictStrNum, Number


def power_for_one_proportion(
    prop_0: Union[DictStrNum, Number, Array],
    prop_1: Union[DictStrNum, Number, Array],
    samp_size: Union[DictStrNum, Number, Array],
    arcsin: bool = True,
    testing_type: str = "two-sided",
    alpha: Union[Number, Array] = 0.05,
) -> Union[DictStrNum, Number, Array]:

    type = testing_type.lower()
    if type not in ("two-sided", "less", "greater"):
        raise AssertionError("type must be 'two-sided', 'less', 'greater'.")

    assert_proportions(prop_0=prop_0, prop_1=prop_1, alpha=alpha)

    if isinstance(alpha, (int, float)):
        z_value = normal().ppf(1 - alpha / 2) if type == "two-sided" else normal().ppf(1 - alpha)
    if isinstance(alpha, (np.ndarray, pd.Series, list, tuple)):
        alpha = numpy_array(alpha)
        z_value = normal().ppf(1 - alpha / 2) if type == "two-sided" else normal().ppf(1 - alpha)

    if isinstance(prop_0, dict) and isinstance(prop_1, dict) and isinstance(samp_size, dict):
        power: dict = {}
        for s in prop_0:
            if arcsin:
                z = (
                    2 * math.asin(math.sqrt(prop_1[s])) - 2 * math.asin(math.sqrt(prop_0[s]))
                ) * math.sqrt(samp_size[s])
            else:
                z = (prop_1[s] - prop_0[s]) / math.sqrt(prop_1[s] * (1 - prop_1[s]) / samp_size[s])

            if isinstance(alpha, dict):
                z_value = (
                    normal().ppf(1 - alpha[s] / 2)
                    if type == "two-sided"
                    else normal().ppf(1 - alpha[s])
                )

            if type == "two-sided":
                power[s] = normal().cdf(abs(z) - z_value)
            elif type == "greater":
                power[s] = normal().cdf(z - z_value)
            else:  # type == "less":
                power[s] = normal().cdf(-z - z_value)
    elif (
        isinstance(prop_0, (int, float))
        and isinstance(prop_1, (int, float))
        and isinstance(samp_size, (int, float))
    ):
        if arcsin:
            z = (2 * math.asin(math.sqrt(prop_1)) - 2 * math.asin(math.sqrt(prop_0))) * math.sqrt(
                samp_size
            )
        else:
            z = (prop_1 - prop_0) / math.sqrt(prop_1 * (1 - prop_1) / samp_size)

        if type == "two-sided":
            power = normal().cdf(abs(z) - z_value)
        elif type == "greater":
            power = normal().cdf(z - z_value)
        else:  # type == "less":
            power = normal().cdf(-z - z_value)
    elif (
        isinstance(prop_0, (np.ndarray, pd.Series, list, tuple))
        and isinstance(prop_1, (np.ndarray, pd.Series, list, tuple))
        and isinstance(samp_size, (np.ndarray, pd.Series, list, tuple))
    ):
        prop_0 = numpy_array(prop_0)
        prop_1 = numpy_array(prop_1)
        samp_size = numpy_array(samp_size)

        for s in prop_0:
            if arcsin:
                z = (2 * np.arcsin(np.sqrt(prop_1)) - 2 * np.arcsin(np.sqrt(prop_0))) * np.sqrt(
                    samp_size
                )
            else:
                z = (prop_1 - prop_0) / np.sqrt(prop_1 * (1 - prop_1) / samp_size)

            if type == "two-sided":
                power = normal().cdf(np.abs(z) - z_value)
            elif type == "greater":
                power = normal().cdf(z - z_value)
            else:  # type == "less":π
                power = normal().cdf(-z - z_value)

    return power


def power_for_two_proportions(
    prop_a: Union[DictStrNum, Number, Array],
    prop_b: Union[DictStrNum, Number, Array],
    samp_size: Optional[Union[DictStrNum, Number, Array]] = None,
    ratio: Optional[Union[DictStrNum, Number, Array]] = None,
    samp_size_a: Optional[Union[DictStrNum, Number, Array]] = None,
    samp_size_b: Optional[Union[DictStrNum, Number, Array]] = None,
    testing_type: str = "two-sided",
    alpha: Union[Number, Array] = 0.05,
) -> Union[DictStrNum, Number, Array]:

    type = testing_type.lower()
    if type not in ("two-sided", "less", "greater"):
        raise AssertionError("type must be 'two-sided', 'less', 'greater'.")

    assert_proportions(prop_a=prop_a, prop_b=prop_b, alpha=alpha)


def power_for_one_mean(
    mean_0: Union[DictStrNum, Number, Array],
    mean_1: Union[DictStrNum, Number, Array],
    sigma: Union[DictStrNum, Number, Array],
    samp_size: Union[DictStrNum, Number, Array],
    testing_type: str = "two-sided",
    alpha: Union[Number, Array] = 0.05,
) -> Union[DictStrNum, Number, Array]:  

    type = testing_type.lower()
    if type not in ("two-sided", "less", "greater"):
        raise AssertionError("type must be 'two-sided', 'less', 'greater'.")

    assert_proportions(alpha=alpha)

    if (
        isinstance(mean_0, dict)
        and isinstance(mean_1, dict)
        and isinstance(sigma, dict)
        and isinstance(samp_size, dict)
    ):
        if type == "two-sided":
            return {
                s: normal().cdf(
                    abs(mean_0[s] - mean_1[s]) / (sigma[s] / math.sqrt(samp_size[s]))
                    - normal().ppf(1 - alpha / 2)
                )
                for s in mean_0
            }
        elif type == "greater":
            return normal().cdf(
                (mean_0 - mean_1) / (sigma / math.sqrt(samp_size)) - normal().ppf(1 - alpha)
            )
        else:
            return normal().cdf(
                -(mean_0 - mean_1) / (sigma / math.sqrt(samp_size)) - normal().ppf(1 - alpha)
            )
    elif (
        isinstance(mean_0, (int, float))
        and isinstance(mean_1, (int, float))
        and isinstance(sigma, (int, float))
        and isinstance(samp_size, (int, float))
    ):
        if type == "two-sided":
            return normal().cdf(
                abs(mean_0 - mean_1) / (sigma / math.sqrt(samp_size)) - normal().ppf(1 - alpha / 2)
            )
        elif type == "greater":
            return normal().cdf(
                (mean_0 - mean_1) / (sigma / math.sqrt(samp_size)) - normal().ppf(1 - alpha)
            )
        else:
            return normal().cdf(
                -(mean_0 - mean_1) / (sigma / math.sqrt(samp_size)) - normal().ppf(1 - alpha)
            )

    elif (
        isinstance(mean_0, (np.np.ndarray, pd.Series, list, tuple))
        and isinstance(mean_1, (np.np.ndarray, pd.Series, list, tuple))
        and isinstance(sigma, (np.np.ndarray, pd.Series, list, tuple))
        and isinstance(samp_size, (np.np.ndarray, pd.Series, list, tuple))
    ):
        mean_0 = numpy_array(mean_0)
        mean_1 = numpy_array(mean_1)
        sigma = numpy_array(sigma)
        power = np.zeros(mean_0.shape[0])
        for k in range(mean_0.shape[0]):
            if type == "two-sided":
                power[k] = normal().cdf(
                    abs(mean_0[k] - mean_1[k]) / (sigma[k] / math.sqrt(samp_size[k]))
                    - normal().ppf(1 - alpha / 2)
                )
            elif type == "greater":
                power[k] = normal().cdf(
                    (mean_0[k] - mean_1[k]) / (sigma[k] / math.sqrt(samp_size[k]))
                    - normal().ppf(1 - alpha)
                )
            else:
                power[k] = normal().cdf(
                    -(mean_0[k] - mean_1[k]) / (sigma[k] / math.sqrt(samp_size[k]))
                    - normal().ppf(1 - alpha)
                )
        return power


# def sample_size_for_proportion_wald(
#     target: Union[DictStrNum, Number, Array],
#     half_ci: Union[DictStrNum, Number, Array],
#     pop_size: Optional[Union[DictStrNum, Number, Array]],
#     deff_c: Union[DictStrNum, Number, Array],
#     alpha: float,
# ) -> Union[DictStrNum, Number, Array]:

#     z_value = normal().ppf(1 - alpha / 2)

#     if isinstance(target, dict) and isinstance(half_ci, dict) and isinstance(deff_c, dict):
#         samp_size: DictStrNum = {}
#         for s in half_ci:
#             sigma_s = target[s] * (1 - target[s])
#             if isinstance(pop_size, dict):
#                 samp_size[s] = math.ceil(
#                     deff_c[s]
#                     * pop_size[s]
#                     * z_value ** 2
#                     * sigma_s
#                     / ((pop_size[s] - 1) * half_ci[s] ** 2 + z_value * sigma_s)
#                 )
#             else:
#                 samp_size[s] = math.ceil(deff_c[s] * z_value ** 2 * sigma_s / half_ci[s] ** 2)
#         return samp_size
#     elif (
#         isinstance(target, (np.ndarray, pd.Series, list, tuple))
#         and isinstance(half_ci, (np.ndarray, pd.Series, list, tuple))
#         and isinstance(deff_c, (np.ndarray, pd.Series, list, tuple))
#     ):
#         target = numpy_array(target)
#         half_ci = numpy_array(half_ci)
#         deff_c = numpy_array(deff_c)
#         samp_size = np.ceil(deff_c * (z_value ** 2) * target * (1 - target) / (half_ci ** 2))
#         return samp_size
#     elif (
#         isinstance(target, (int, float))
#         and isinstance(half_ci, (int, float))
#         and isinstance(deff_c, (int, float))
#     ):
#         sigma = target * (1 - target)
#         if isinstance(pop_size, (int, float)):
#             return math.ceil(
#                 deff_c
#                 * pop_size
#                 * z_value ** 2
#                 * sigma
#                 / ((pop_size - 1) * half_ci ** 2 + z_value * sigma)
#             )
#         else:
#             return math.ceil(deff_c * z_value ** 2 * sigma / half_ci ** 2)
#     else:
#         raise TypeError("target and half_ci must be numbers or dictionaries!")


# def sample_size_for_proportion_fleiss(
#     target: Union[DictStrNum, Number, Array],
#     half_ci: Union[DictStrNum, Number, Array],
#     deff_c: Union[DictStrNum, Number, Array],
#     alpha: float,
# ) -> Union[DictStrNum, Number, Array]:

#     z_value = normal().ppf(1 - alpha / 2)

#     def fleiss_factor(p: float, d: float) -> float:

#         if 0 <= p < d or 1 - d < p <= 1:
#             return 8 * d * (1 - 2 * d)
#         elif d <= p < 0.3:
#             return 4 * (p + d) * (1 - p - d)
#         elif 0.7 < p <= 1 - d:
#             return 4 * (p - d) * (1 - p + d)
#         elif 0.3 <= p <= 0.7:
#             return 1
#         else:
#             raise ValueError("Parameters p or d not valid.")

#     if isinstance(target, dict) and isinstance(half_ci, dict) and isinstance(deff_c, dict):
#         samp_size: DictStrNum = {}
#         for s in half_ci:
#             fct = fleiss_factor(target[s], half_ci[s])
#             samp_size[s] = math.ceil(
#                 deff_c[s]
#                 * (
#                     fct * (z_value ** 2) / (4 * half_ci[s] ** 2)
#                     + 1 / half_ci[s]
#                     - 2 * z_value ** 2
#                     + (z_value + 2) / fct
#                 )
#             )
#         return samp_size
#     elif (
#         isinstance(target, (np.ndarray, pd.Series, list, tuple))
#         and isinstance(half_ci, (np.ndarray, pd.Series, list, tuple))
#         and isinstance(deff_c, (np.ndarray, pd.Series, list, tuple))
#     ):
#         target = numpy_array(target)
#         half_ci = numpy_array(half_ci)
#         deff_c = numpy_array(deff_c)
#         samp_size = np.zeros(target.shape[0])
#         for k in range(target.shape[0]):
#             fct = fleiss_factor(target[k], half_ci[k])
#             samp_size[k] = math.ceil(
#                 deff_c[k]
#                 * (
#                     fct * (z_value ** 2) / (4 * half_ci[k] ** 2)
#                     + 1 / half_ci[k]
#                     - 2 * z_value ** 2
#                     + (z_value + 2) / fct
#                 )
#             )
#         return samp_size
#     elif (
#         isinstance(target, (int, float))
#         and isinstance(half_ci, (int, float))
#         and isinstance(deff_c, (int, float))
#     ):
#         fct = fleiss_factor(target, half_ci)
#         return math.ceil(
#             deff_c
#             * (
#                 fct * (z_value ** 2) / (4 * half_ci ** 2)
#                 + 1 / half_ci
#                 - 2 * z_value ** 2
#                 + (z_value + 2) / fct
#             )
#         )
#     else:
#         raise TypeError("target and half_ci must be numbers or dictionaries!")


# def sample_size_for_mean_wald(
#     half_ci: Union[DictStrNum, Number, Array],
#     sigma: Union[DictStrNum, Number, Array],
#     pop_size: Optional[Union[DictStrNum, Number, Array]],
#     deff_c: Union[DictStrNum, Number, Array],
#     alpha: float,
# ) -> Union[DictStrNum, Number, Array]:

#     z_value = normal().ppf(1 - alpha / 2)

#     if isinstance(half_ci, dict) and isinstance(sigma, dict) and isinstance(deff_c, dict):
#         samp_size: DictStrNum = {}
#         for s in half_ci:
#             if isinstance(pop_size, dict):
#                 samp_size[s] = math.ceil(
#                     deff_c[s]
#                     * pop_size[s]
#                     * z_value ** 2
#                     * sigma[s] ** 2
#                     / ((pop_size[s] - 1) * half_ci[s] ** 2 + z_value ** 2 * sigma[s] ** 2)
#                 )
#             else:
#                 samp_size[s] = math.ceil(deff_c[s] * (z_value * sigma[s] / half_ci[s]) ** 2)
#         return samp_size
#     elif (
#         isinstance(half_ci, (np.ndarray, pd.Series, list, tuple))
#         and isinstance(sigma, (np.ndarray, pd.Series, list, tuple))
#         and isinstance(deff_c, (np.ndarray, pd.Series, list, tuple))
#     ):
#         half_ci = numpy_array(half_ci)
#         sigma = numpy_array(sigma)
#         deff_c = numpy_array(deff_c)
#         return np.ceil(deff_c * (z_value * sigma / half_ci) ** 2)
#     elif (
#         isinstance(half_ci, (int, float))
#         and isinstance(sigma, (int, float))
#         and isinstance(deff_c, (int, float))
#     ):
#         if isinstance(pop_size, (int, float)):
#             return math.ceil(
#                 deff_c
#                 * pop_size
#                 * z_value ** 2
#                 * sigma ** 2
#                 / ((pop_size - 1) * half_ci ** 2 + z_value ** 2 * sigma ** 2)
#             )
#         else:
#             return math.ceil(deff_c * (z_value * sigma / half_ci) ** 2)
#     else:
#         raise TypeError("target, half_ci, and sigma must be numbers or dictionaries!")
