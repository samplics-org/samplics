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


def _calculate_ss_wald_mean_one_sample(
    two_sides: bool,
    epsilon: Union[Array, Number],
    delta: Union[Array, Number],
    sigma: Union[Array, Number],
    deff_c: Union[Array, Number],
    alpha: Union[Array, Number],
    power: Union[Array, Number],
) -> Union[Array, Number]:

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


def _calculate_ss_wald_mean_one_sample_stratified(
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
        samp_size[s] = _calculate_ss_wald_mean_one_sample(
            two_sides=two_sides[s],
            epsilon=epsilon[s],
            delta=delta[s],
            sigma=sigma[s],
            deff_c=deff_c[s],
            alpha=alpha[s],
            power=power[s],
        )
    return samp_size


def calculate_ss_wald_mean_one_sample(
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
        return _calculate_ss_wald_mean_one_sample_stratified(
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
        return _calculate_ss_wald_mean_one_sample(
            two_sides=two_sides,
            epsilon=epsilon,
            delta=delta,
            sigma=sigma,
            deff_c=deff_c,
            alpha=alpha,
            power=power,
        )


def _calculate_ss_wald_mean_two_sample(
    two_sides: bool,
    epsilon: Union[Array, Number],
    delta: Union[Array, Number],
    sigma_1: Union[Array, Number],
    sigma_2: Optional[Union[Array, Number]],
    equal_variance: bool,
    kappa: Union[Array, Number],
    deff_c: Union[Array, Number],
    alpha: Union[Array, Number],
    power: Union[Array, Number],
) -> tuple[Union[Array, Number], Union[Array, Number]]:

    if two_sides and delta == 0:
        z_alpha = normal().ppf(1 - alpha / 2)
        z_beta = normal().ppf(power)
    elif two_sides and delta != 0:
        z_alpha = normal().ppf(1 - alpha)
        z_beta = normal().ppf((1 + power) / 2)  # 1 - beta/2 where beta = 1 - power
    else:
        z_alpha = normal().ppf(1 - alpha)
        z_beta = normal().ppf(power)

    if equal_variance:
        samp_size_2 = math.ceil(
            deff_c * (1 + 1 / kappa) * ((z_alpha + z_beta) * sigma_1 / (delta - abs(epsilon))) ** 2
        )
        samp_size_1 = math.ceil(kappa * samp_size_2)
    else:
        pass

    return (samp_size_1, samp_size_2)


def _calculate_ss_wald_mean_two_sample_stratified(
    two_sides: bool,
    epsilon: Union[Array, Number],
    delta: Union[Array, Number],
    sigma_1: Union[Array, Number],
    sigma_2: Optional[Union[Array, Number]],
    equal_variance: bool,
    kappa: Union[Array, Number],
    deff_c: Union[Array, Number],
    alpha: Union[Array, Number],
    power: Union[Array, Number],
) -> tuple[DictStrNum, DictStrNum]:

    samp_size_1: DictStrNum = {}
    samp_size_2: DictStrNum = {}
    for s in delta:
        sigma_2_s = sigma_2[s] if sigma_2 is not None else None
        samp_size_1[s], samp_size_2[s] = _calculate_ss_wald_mean_one_sample(
            two_sides=two_sides[s],
            epsilon=epsilon[s],
            delta=delta[s],
            sigma_1=sigma_1[s],
            sigma_2=sigma_2_s,
            equal_variance=equal_variance,
            kappa=kappa,
            deff_c=deff_c[s],
            alpha=alpha[s],
            power=power[s],
        )
    return (samp_size_1, samp_size_2)


def calculate_ss_wald_mean_two_sample(
    two_sides: bool,
    epsilon: Union[DictStrNum, Number, Array],
    delta: Union[DictStrNum, Number, Array],
    sigma_1: Union[DictStrNum, Number, Array],
    sigma_2: Union[DictStrNum, Number, Array],
    equal_variance: Union[DictStrNum, Number, Array],
    kappa: Union[DictStrNum, Number, Array],
    deff_c: Union[DictStrNum, Number, Array],
    alpha: Union[DictStrNum, Number, Array],
    power: Union[DictStrNum, Number, Array],
    stratification: bool = True,
) -> DictStrNum:

    if stratification:
        return _calculate_ss_wald_mean_two_sample_stratified(
            two_sides=two_sides,
            epsilon=epsilon,
            delta=delta,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            equal_variance=equal_variance,
            kappa=kappa,
            deff_c=deff_c,
            alpha=alpha,
            power=power,
            stratification=stratification,
        )
    else:
        return _calculate_ss_wald_mean_two_sample(
            two_sides=two_sides,
            epsilon=epsilon,
            delta=delta,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            equal_variance=equal_variance,
            kappa=kappa,
            deff_c=deff_c,
            alpha=alpha,
            power=power,
        )


def _calculate_ss_wald_prop_two_sample(
    two_sides: bool,
    epsilon: Union[Array, Number],
    prop_1: Union[Array, Number],
    prop_2: Union[Array, Number],
    delta: Union[Array, Number],
    kappa: Union[Array, Number],
    deff_c: Union[Array, Number],
    alpha: Union[Array, Number],
    power: Union[Array, Number],
) -> tuple[Union[Array, Number], Union[Array, Number]]:

    if two_sides and delta == 0:
        z_alpha = normal().ppf(1 - alpha / 2)
        z_beta = normal().ppf(power)
    elif two_sides and delta != 0:
        z_alpha = normal().ppf(1 - alpha)
        z_beta = normal().ppf((1 + power) / 2)  # 1 - beta/2 where beta = 1 - power
    else:
        z_alpha = normal().ppf(1 - alpha)
        z_beta = normal().ppf(power)

    samp_size_2 = math.ceil(
        deff_c
        * (prop_1 * (1 - prop_1) / kappa + prop_2 * (1 - prop_2))
        * ((z_alpha + z_beta) / (delta - abs(epsilon))) ** 2
    )
    samp_size_1 = math.ceil(kappa * samp_size_2)

    return (samp_size_1, samp_size_2)


def _calculate_ss_wald_prop_two_sample_stratified(
    two_sides: bool,
    epsilon: DictStrNum,
    prop_1: DictStrNum,
    prop_2: DictStrNum,
    delta: DictStrNum,
    kappa: DictStrNum,
    deff_c: DictStrNum,
    alpha: DictStrNum,
    power: DictStrNum,
) -> DictStrNum:

    samp_size: DictStrNum = {}
    for s in delta:
        samp_size[s] = _calculate_ss_wald_prop_two_sample(
            two_sides=two_sides[s],
            epsilon=epsilon[s],
            prop_1=prop_1[s],
            prop_2=prop_2[s],
            delta=delta[s],
            kappa=kappa[s],
            deff_c=deff_c[s],
            alpha=alpha[s],
            power=power[s],
        )
    return samp_size


def calculate_ss_wald_prop_two_sample(
    two_sides: bool,
    epsilon: Union[DictStrNum, Number, Array],
    prop_1: Union[DictStrNum, Number, Array],
    prop_2: Union[DictStrNum, Number, Array],
    delta: Union[DictStrNum, Number, Array],
    kappa: Union[DictStrNum, Number, Array],
    deff_c: Union[DictStrNum, Number, Array],
    alpha: Union[DictStrNum, Number, Array],
    power: Union[DictStrNum, Number, Array],
    stratification: bool = True,
) -> DictStrNum:

    if stratification:
        return _calculate_ss_wald_prop_two_sample_stratified(
            two_sides=two_sides,
            epsilon=epsilon,
            prop_1=prop_1,
            prop_2=prop_2,
            delta=delta,
            kappa=kappa,
            deff_c=deff_c,
            alpha=alpha,
            power=power,
            stratification=stratification,
        )
    else:
        return _calculate_ss_wald_prop_two_sample(
            two_sides=two_sides,
            epsilon=epsilon,
            prop_1=prop_1,
            prop_2=prop_2,
            delta=delta,
            kappa=kappa,
            deff_c=deff_c,
            alpha=alpha,
            power=power,
        )
