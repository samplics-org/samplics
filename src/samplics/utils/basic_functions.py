"""This module provides basic functions used across multiple classes and modules. 

Functions:
    | *sumby()* sums numpy arrays by some grouping. 
    | *averageby()* average numpy arrays by some grouping.
    | *transform()* implements the Boxcox tranformation.
    | *skewness()* computes the Pearson's moment coefficient of skewness (assymetry measure). 
    | *kurtosis()* computes the fourth standardized moment, a measure of assymetry.
    | *plot_skewness()* and *plot_kurtosis()* visualise the skewness and kurtosis coefficients 
respectively. 

"""
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
    """Sums the numpy array by group. 

    Args:
        group (np.ndarray): provides the group associated to each observation. 
        y (np.ndarray): variable of interest for the summation. 

    Returns:
        np.ndarray: the sums by group. 
    """

    groups = np.unique(group)
    sums = np.zeros(groups.size)
    for k, gr in enumerate(groups):
        sums[k] = np.sum(y[group == gr])

    return sums


def averageby(
    group: np.ndarray, y: np.ndarray
) -> np.ndarray:  # Could use pd.grouby().sum(), may scale better
    """[summary]

    Args:
        group (np.ndarray): provides the group associated to each observation. 
        y (np.ndarray): variable of interest for the averaging. 

    Returns:
        np.ndarray: the averages by group. 
    """

    groups = np.unique(group)
    means = np.zeros(groups.size)
    for k, gr in enumerate(groups):
        means[k] = np.mean(y[group == gr])

    return means


def transform(
    y: np.ndarray, llambda: Optional[Number] = None, constant: float = 0.0, inverse: bool = True,
) -> np.ndarray:
    """Transforms the variable of interest using the Boxcox method or its inverse. 

    Args:
        y (np.ndarray): variable of interest. 
        llambda (Optional[Number], optional): Boxcox parameter lambda. A value of zero indicates
        a logarithm transformation. Defaults to None.
        constant (float, optional): An additive term to ensure that the logarithm function is 
        applied to positive values. Defaults to 0.0.
        inverse (bool, optional): indicates if the Boxcox transformation is applied or its 
        inverse. Defaults to True.

    Returns:
        np.ndarray: [description]
    """
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


def _plot_measure(
    y: np.ndarray,
    coef_min: Number = -5,
    coef_max: Number = 5,
    nb_points: int = 100,
    measure: str = "skewness",
) -> None:
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


def plot_skewness(
    y: np.ndarray, coef_min: Number = -5, coef_max: Number = 5, nb_points: int = 100,
) -> None:
    """Plots a scatter plot of skewness coefficients and the lambda coefficients of the Boxcox
    transformation. The plot helps identify the range of lambda values to minimize the 
    skewness coefficient. 

    Args:
        y (np.ndarray): variable of interest. 
        coef_min (Number, optional): lowest value for the horizontal axis. Defaults to -5.
        coef_max (Number, optional): highest value for the horizontal axis. Defaults to 5.
        nb_points (int, optional): number of points in the plot. Defaults to 100.
    """
    _plot_measure(
        y=y, coef_min=coef_min, coef_max=coef_max, nb_points=nb_points, measure="skewness",
    )


def plot_kurtosis(
    y: np.ndarray, coef_min: Number = -5, coef_max: Number = 5, nb_points: int = 100,
) -> None:
    """Plots a scatter plot of skewness coefficients and the lambda coefficients of the Boxcox
    transformation. The plot helps identify the range of lambda values to minimize the 
    kurtosis coefficient. 

    Args:
        y (np.ndarray): variable of interest. 
        coef_min (Number, optional): lowest value for the horizontal axis. Defaults to -5.
        coef_max (Number, optional): highest value for the horizontal axis. Defaults to 5.
        nb_points (int, optional): number of points in the plot. Defaults to 100.
    """
    _plot_measure(
        y=y, coef_min=coef_min, coef_max=coef_max, nb_points=nb_points, measure="kurtosis",
    )
