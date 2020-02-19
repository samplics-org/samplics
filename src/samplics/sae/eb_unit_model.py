from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import math

import statsmodels.api as sm

from scipy.stats import boxcox, norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


class UnitModel:
    """implements the unit level model"""

    def __init__(
        self,
        method: str = "REML",
        parameter: str = "mean",
        boxcox: Optional[float] = None,
        function=None,
    ):
        self.model = "BHF"
        self.method = method.upper()
        self.parameter = parameter.lower()
        self.boxcox = boxcox

        self.fixed_effects: np.ndarray = np.array([])
        self.fe_cov: np.ndarray = np.array([])
        self.random_effects: np.ndarray = np.array([])
        self.re_cov: np.ndarray = np.array([])
        self.convergence: Dict[str, Union[float, int, bool]] = {}
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC
        self.point_est: Dict[Any, float] = {}
        self.mse: Dict[Any, float] = {}
        self.mse_as1: Dict[Any, float] = {}
        self.mse_as2: Dict[Any, float] = {}
        self.mse_terms: Dict[str, Dict[Any, float]] = {}

    def _beta(
        self,
        y: np.ndarray,
        X: np.ndarray,
        area: np.ndarray,
        weight: np.ndarray,
        e_scale: np.ndarray,
    ) -> np.ndarray:

        y = formats.numpy_array(y)
        X = formats.numpy_array(X)
        weight = formats.numpy_array(weight)
        area = formats.numpy_array(area)

        Xw = X * weight[:, None]
        p = X.shape[1]
        beta1 = np.zeros((p, p))
        beta2 = np.zeros(p)
        for k, d in enumerate(np.unique(area)):
            w_d = weight[area == d]
            y_d = y[area == d]
            X_d = X[area == d]
            Xw_d = Xw[area == d]
            Xw_d_bar = np.sum(Xw_d, axis=0) / np.sum(w_d)
            resid_d_w = X_d - Xw_d_bar * self.gamma[k]
            beta1 = beta1 + np.matmul(np.transpose(Xw_d), resid_d_w)
            beta2 = beta2 + np.sum(resid_d_w * y_d[:, None] * w_d[:, None], axis=0)

        beta = np.matmul(np.linalg.inv(beta1), beta2)

        return beta

    def _area_stats(self, arr1, arr2, area, samp_weight, scale):

        arr1 = formats.numpy_array(arr1)
        arr2 = formats.numpy_array(arr2)

        if samp_weight is None:
            weight = np.ones(arr1.size)

        a_factor = 1 / (scale ** 2)

        areas = np.unique(area)
        arr1_mean = np.zeros(areas.size)
        arr2_mean = np.zeros((areas.size, arr2.shape[1]))
        gamma = np.zeros(areas.size)
        for k, d in enumerate(areas):
            a_factor_d = a_factor[area == d]
            weight_d = weight[area == d]
            aw_factor_d = weight_d * a_factor_d
            arr1w_d = arr1[area == d] * aw_factor_d
            arr1_mean[k] = np.sum(arr1w_d) / np.sum(aw_factor_d)
            arr2w_d = arr2[area == d, :] * aw_factor_d[:, None]
            arr2_mean[k, :] = np.sum(arr2w_d, axis=0) / np.sum(aw_factor_d)
            if samp_weight is None:
                delta_d = 1 / np.sum(a_factor_d)
            else:
                delta_d = np.sum((weight_d / np.sum(weight_d)) ** 2)
            gamma[k] = self.re_cov / (self.re_cov + self.sigma2_e * delta_d)

        return arr1_mean, arr2_mean, gamma

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        area: np.ndarray,
        samp_weight: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
    ) -> None:

        if intercept and isinstance(X, np.ndarray):
            X = np.insert(X, 0, 1, axis=1)
        elif intercept and isinstance(X, pd.DataFrame):
            X.insert(0, "Intercept", 1, False)

        if samp_weight is not None and isinstance(samp_weight, pd.DataFrame):
            samp_weight = formats.numpy_array(samp_weight)

        if isinstance(scale, (float, int)):
            scale = np.ones(y.shape[0]) * scale

        reml = True if self.method == "REML" else False
        basic_model = sm.MixedLM(y, X, area)
        basic_fit = basic_model.fit(reml=reml, full_output=True)
        self.area_s = np.unique(formats.numpy_array(area))

        self.sigma2_e = basic_fit.scale
        self.fixed_effects = basic_fit.fe_params
        self.random_effects = basic_fit.cov_re.to_numpy()

        self.fe_cov = basic_fit.bse_fe
        self.re_cov = basic_fit.cov_re.to_numpy()
        self.re_cov_cov = basic_fit.bse_re
        self.convergence["achieved"] = basic_fit.converged
        self.convergence["iterations"] = len(basic_fit.hist[0]["allvecs"])

        nb_obs = y.shape[0]
        nb_variance_params = basic_fit.cov_re.shape[0] + 1
        if self.method == "REML":  # page 111 - Rao and Molina (2015)
            aic = -2 * basic_fit.llf + 2 * nb_variance_params
            bic = (
                -2 * basic_fit.llf
                + np.log(nb_obs - self.fixed_effects.shape[0]) * nb_variance_params
            )
        elif self.method == "ML":
            aic = -2 * basic_fit.llf + 2 * (self.fixed_effects.shape[0] + nb_variance_params)
            bic = -2 * basic_fit.llf + np.log(nb_obs) * (
                self.fixed_effects.shape[0] + nb_variance_params
            )
        else:
            aic = np.nan
            bic = np.nan
        self.goodness["loglike"] = basic_fit.llf
        self.goodness["AIC"] = aic
        self.goodness["BIC"] = bic

        self.ybar_s, self.Xbar_s, self.gamma = self._area_stats(y, X, area, samp_weight, scale)

        # samp_weight = np.ones(y.size)
        if samp_weight is not None:
            beta_w = self._beta(y, X, area, samp_weight, scale)
            # print(beta_w)

    def _predictor_basic(
        self, y_s: Array, X_s: Array, X_r: Array, X_smean: Array, X_rmean: Array, area: np.ndarray
    ) -> np.ndarray:

        term1 = np.matmul(X_smean, self.fixed_effects)
        term2 = self.gamma * ()

    def predict(self, X: Array, area: Array, intercept: bool = True,) -> None:

        X = formats.numpy_array(X)
        if intercept:
            X = np.insert(X, 0, 1, axis=1)

        random_effect = self.gamma * (self.ybar_s - np.matmul(self.Xbar_s, self.fixed_effects))

        y_predict = np.matmul(X, self.fixed_effects) + random_effect

        self.random_effect = random_effect
        self.y_predicted = y_predict


class UnitModelRobust:
    """implement the robust unit level model"""

    pass
