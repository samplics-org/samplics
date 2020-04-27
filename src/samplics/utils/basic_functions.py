from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.stats import boxcox, norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


def sumby(group, y):  # Could use pd.grouby().sum(), may scale better

    groups = np.unique(group)
    sums = np.zeros(groups.size)
    for k, gr in enumerate(groups):
        sums[k] = np.sum(y[group == gr])

    return sums


class BoxCox:
    def get_skewness(self, y: Array) -> float:

        y = formats.numpy_array(y)
        self.skewness = float(np.mean((y - np.mean(y)) ** 3) / np.std(y) ** 3)

        return self.skewness

    def get_kurtosis(self, y: Array) -> float:

        y = formats.numpy_array(y)
        self.kurtosis = float(np.mean((y - np.mean(y)) ** 4) / np.std(y) ** 4 - 3)

        return self.kurtosis

    def transform(self, y: Array, coef: Number) -> np.ndarray:

        y = formats.numpy_array(y)

        if np.min(y) <= 0:
            y = y - np.min(y) + 1

        if coef == 0:
            y_transformed = np.log(y)
        else:
            y_transformed = (np.power(y, coef) - 1) / coef

        return y_transformed

    def _plot_measure(self, y, coef_min=-5, coef_max=5, nb_points=100, measure="skewness") -> None:
        y = formats.numpy_array(y)
        lambda_range = np.linspace(coef_min, coef_max, num=nb_points)
        coefs = np.zeros(lambda_range.size)
        for k, ll in enumerate(lambda_range):
            y_ll = self.transform(y, ll)
            if measure.lower() == "skewness":
                coefs[k] = self.get_skewness(y_ll)
                measure_loc = "lower right"
            elif measure.lower() == "kurtosis":
                coefs[k] = self.get_kurtosis(y_ll)
                measure_loc = "upper right"

        normality = np.abs(coefs) < 2.0

        p1 = plt.scatter(
            lambda_range[normality], coefs[normality], marker="D", c="green", s=25, alpha=0.3,
        )
        p2 = plt.scatter(
            lambda_range[~normality],
            coefs[~normality],
            c="red",
            s=10,
            alpha=0.6,
            edgecolors="none",
        )
        plt.axhline(0, color="blue", linestyle="--")
        plt.title(f"{measure.title()} by BoxCox lambda")
        plt.ylabel(f"{measure.title()}")
        plt.xlabel("Lambda (coefs)")
        legent = plt.legend((p1, p2), ("Normality zone", "Non-normality zone"), loc=measure_loc,)
        plt.show()

    def plot_skewness(self, y, coef_min=-5, coef_max=5, nb_points=100) -> None:
        self._plot_measure(
            y=y, coef_min=coef_min, coef_max=coef_max, nb_points=nb_points, measure="skewness"
        )

    def plot_kurtosis(self, y, coef_min=-5, coef_max=5, nb_points=100) -> None:
        self._plot_measure(
            y=y, coef_min=coef_min, coef_max=coef_max, nb_points=nb_points, measure="kurtosis"
        )
