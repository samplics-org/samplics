from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.stats import boxcox, norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


def sumby(
    group: np.ndarray, y: np.ndarray
) -> np.ndarray:  # Could use pd.grouby().sum(), may scale better

    groups = np.unique(group)
    sums = np.zeros(groups.size)
    for k, gr in enumerate(groups):
        sums[k] = np.sum(y[group == gr])

    return sums


def averageby(
    group: np.ndarray, y: np.ndarray
) -> np.ndarray:  # Could use pd.grouby().sum(), may scale better

    groups = np.unique(group)
    means = np.zeros(groups.size)
    for k, gr in enumerate(groups):
        means[k] = np.mean(y[group == gr])

    return means


def transform(
    y: np.ndarray, llambda: Optional[Number] = None, constant: float = 0.0, inverse: bool = True,
) -> np.ndarray:
    if llambda is None:
        z = y
    elif llambda == 0.0:
        if inverse:
            z = np.exp(y) - constant
        else:
            z = np.log(y + constant)
    elif boxcox["lambda"] != 0.0:
        if inverse:
            z = np.exp(np.log(1 + y * llambda) / llambda)
        else:
            z = np.power(y, llambda) / llambda
    return z


def skewness(y: Array) -> float:

    y = formats.numpy_array(y)
    skewness = float(np.mean((y - np.mean(y)) ** 3) / np.std(y) ** 3)

    return skewness


def kurtosis(y: Array) -> float:

    y = formats.numpy_array(y)
    kurtosis = float(np.mean((y - np.mean(y)) ** 4) / np.std(y) ** 4 - 3)

    return kurtosis


def _plot_measure(y, coef_min=-5, coef_max=5, nb_points=100, measure="skewness") -> None:
    y = formats.numpy_array(y)
    lambda_range = np.linspace(coef_min, coef_max, num=nb_points)
    coefs = np.zeros(lambda_range.size)
    for k, ll in enumerate(lambda_range):
        y_ll = transform(y, ll)
        if measure.lower() == "skewness":
            coefs[k] = skewness(y_ll)
            measure_loc = "lower right"
        elif measure.lower() == "kurtosis":
            coefs[k] = kurtosis(y_ll)
            measure_loc = "upper right"

    normality = np.abs(coefs) < 2.0

    p1 = plt.scatter(
        lambda_range[normality], coefs[normality], marker="D", c="green", s=25, alpha=0.3,
    )
    p2 = plt.scatter(
        lambda_range[~normality], coefs[~normality], c="red", s=10, alpha=0.6, edgecolors="none",
    )
    plt.axhline(0, color="blue", linestyle="--")
    plt.title(f"{measure.title()} by BoxCox lambda")
    plt.ylabel(f"{measure.title()}")
    plt.xlabel("Lambda (coefs)")
    legent = plt.legend((p1, p2), ("Normality zone", "Non-normality zone"), loc=measure_loc,)
    plt.show()


def plot_skewness(y, coef_min=-5, coef_max=5, nb_points=100) -> None:
    _plot_measure(
        y=y, coef_min=coef_min, coef_max=coef_max, nb_points=nb_points, measure="skewness"
    )


def plot_kurtosis(y, coef_min=-5, coef_max=5, nb_points=100) -> None:
    _plot_measure(
        y=y, coef_min=coef_min, coef_max=coef_max, nb_points=nb_points, measure="kurtosis"
    )
