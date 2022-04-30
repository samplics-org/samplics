"""Sample size calculation module 
This module compiles the core functions for calculating power and sample size
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


def _calculate_ss_wald_mean(
    two_sides: bool,
    epsilon: Union[Array, Number],
    delta: Union[Array, Number],
    sigma: Union[Array, Number],
    deff_c: Union[Array, Number],
    alpha: Union[Array, Number],
    power: Union[Array, Number],
) -> Union[DictStrNum, Number]:

    if two_sides and delta == 0:
        z_alpha = normal().ppf(1 - alpha / 2)
        z_beta = normal().ppf(power)
    elif two_sides and delta != 0:
        z_alpha = normal().ppf(1 - alpha)
        z_beta = normal().ppf((1 + power) / 2)  # 1 - beta/2 where beta = 1 - power
    else:
        z_alpha = normal().ppf(1 - alpha)
        z_beta = normal().ppf(power)

    return math.ceil(deff_c * ((z_alpha + z_beta) * sigma / (delta - abs(epsilon))) ** 2)


def _calculate_ss_wald_mean_stratified(
    two_sides: bool,
    epsilon: DictStrNum,
    delta: DictStrNum,
    sigma: DictStrNum,
    deff_c: DictStrNum,
    alpha: DictStrNum,
    power: DictStrNum,
) -> DictStrNum:

    samp_size: DictStrNum = {}
    for s in delta:
        samp_size[s] = _calculate_ss_wald_mean(
            two_sides=two_sides[s],
            epsilon=epsilon[s],
            delta=delta[s],
            sigma=sigma[s],
            deff_c=deff_c[s],
            alpha=alpha[s],
            power=power[s],
        )
    return samp_size


def calculate_ss_wald_mean(
    two_sides: bool,
    epsilon: Union[DictStrNum, Number, Array],
    delta: Union[DictStrNum, Number, Array],
    sigma: Union[DictStrNum, Number, Array],
    deff_c: Union[DictStrNum, Number, Array],
    alpha: Union[DictStrNum, Number, Array],
    power: Union[DictStrNum, Number, Array],
    stratification: bool = True,
) -> DictStrNum:

    if stratification:
        return _calculate_ss_wald_mean_stratified(
            two_sides=two_sides,
            epsilon=epsilon,
            delta=delta,
            sigma=sigma,
            deff_c=deff_c,
            alpha=alpha,
            power=power,
            stratification=stratification,
        )
    else:
        return _calculate_ss_wald_mean(
            two_sides=two_sides,
            epsilon=epsilon,
            delta=delta,
            sigma=sigma,
            deff_c=deff_c,
            alpha=alpha,
            power=power,
        )



