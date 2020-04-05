from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import math

import statsmodels.api as sm

from scipy.stats import boxcox, norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


class EblupUnitLevel:
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

        self.area_s: np.ndarray = np.array([])
        self.area_p: np.ndarray = np.array([])
        self.samp_size = np.array([])
        self.pop_size = np.array([])

        self.fitted = False
        self.fixed_effects: np.ndarray = np.array([])
        self.fe_std: np.ndarray = np.array([])
        self.random_effects: np.ndarray = np.array([])
        self.re_std: Optional[float] = None
        self.re_std_cov: Optional[float] = None
        self.error_std: Optional[float] = None
        self.convergence: Dict[str, Union[float, int, bool]] = {}
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC

        self.ybar_s: np.ndarray = np.array([])
        self.xbar_s: np.ndarray = np.array([])
        self.Xbar_p: np.ndarray = np.array([])
        self.gamma: np.ndarray = np.array([])
        self.scale: np.ndarray = np.array([])

        self.y_predicted: np.ndarray = np.array([])
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
        samp_size = np.zeros(areas.size)
        for k, d in enumerate(areas):
            sample_d = area == d
            a_factor_d = a_factor[sample_d]
            weight_d = weight[sample_d]
            aw_factor_d = weight_d * a_factor_d
            arr1w_d = arr1[sample_d] * a_factor_d
            arr1_mean[k] = np.sum(arr1w_d) / np.sum(aw_factor_d)
            arr2w_d = arr2[sample_d, :] * aw_factor_d[:, None]
            arr2_mean[k, :] = np.sum(arr2w_d, axis=0) / np.sum(aw_factor_d)
            if samp_weight is None:
                delta_d = 1 / np.sum(a_factor_d)
            else:
                delta_d = np.sum((weight_d / np.sum(weight_d)) ** 2)
            gamma[k] = (self.re_std ** 2) / (self.re_std ** 2 + (self.error_std ** 2) * delta_d)
            samp_size[k] = np.sum(sample_d)

        return arr1_mean, arr2_mean, gamma, samp_size

    def _g1(self, gamma: np.ndarray, scale: np.ndarray,) -> np.ndarray:

        # print(self.error_std)
        # print(scale)
        # print(gamma)

        return gamma * (self.error_std ** 2 / scale)

    def _A_matrix(self, area: np.ndarray, X: np.ndarray):

        areas = np.unique(area)
        A = np.diag(np.zeros(X.shape[1]))
        for d in areas:
            n_d = np.sum(area == d)
            X_d = X[area == d]
            V = (self.error_std ** 2) * np.diag(np.ones(n_d)) + (self.re_std ** 2) * np.ones(
                [n_d, n_d]
            )
            A = A + np.matmul(np.matmul(np.transpose(X_d), np.linalg.inv(V)), X_d)

        return A

    @staticmethod
    def _g2(
        areas: np.ndarray,
        Xs_mean: np.ndarray,
        Xp_mean: np.ndarray,
        gamma: np.ndarray,
        A_inv: np.ndarray,
    ) -> np.ndarray:

        xbar_diff = Xp_mean - gamma[:, None] * Xs_mean
        g2_matrix = np.matmul(np.matmul(xbar_diff, A_inv), np.transpose(xbar_diff))

        return np.diag(g2_matrix)

    @staticmethod
    def _g3(
        sigma2u: float, sigma2e: float, scale: np.ndarray, samp_size: np.ndarray
    ) -> np.ndarray:

        alpha = sigma2e + scale * sigma2u

        i_vv = 0.5 * sum((scale / alpha) ** 2)
        i_ee = 0.5 * sum((samp_size - 1) / (sigma2e ** 2) + 1 / (alpha ** 2))
        i_ve = 0.5 * sum(scale / (alpha ** 2))

        i_determinant = i_vv * i_ee - i_ve * i_ve

        g3_scale = 1 / ((scale ** 2) * ((sigma2u + sigma2e / scale) ** 3))
        g3 = g3_scale * (
            (sigma2e ** 2) * (i_ee / i_determinant)
            + (sigma2u ** 2) * (i_vv / i_determinant)
            - 2 * (sigma2e * sigma2u) * (-i_vv / i_determinant)
        )

        return g3

    def _mse1(self, scale: np.ndarray, A_inv: np.ndarray) -> np.ndarray:

        g1 = self._g1(scale)
        g2 = self._g2(A_inv)
        g3 = 0

        return g1 + g2 + 2 * g3

    @staticmethod
    def _sumby(group, y):  # Could use pd.grouby().sum(), may scale better

        groups = np.unique(group)
        sums = np.zeros(groups.size)
        for k, gr in enumerate(groups):
            sums[k] = np.sum(y[group == gr])

        return sums

    def _predict(
        self,
        pop_size: np.ndarray,
        Xmean_ps: np.ndarray,
        Xmean_pr: np.ndarray,
        xbar_ps: np.ndarray,
        gamma_ps: np.ndarray,
        samp_rate_ps: np.ndarray,
        ps: np.ndarray,
        ps_area: np.ndarray,
    ) -> np.ndarray:

        if pop_size is None or pop_size is None:
            self.fpc = np.zeros(Xmean_ps.shape[0])
            ys_pred = np.matmul(Xmean_ps, self.fixed_effects) + gamma_ps
        else:
            ys_pred = np.matmul(Xmean_ps, self.fixed_effects) + (
                samp_rate_ps + (1 - samp_rate_ps) * gamma_ps
            ) * (self.ybar_s[ps_area] - np.matmul(xbar_ps, self.fixed_effects))

        yr_pred = np.array([])
        if np.sum(~ps) > 0:
            yr_pred = np.matmul(Xmean_pr, self.fixed_effects)

        return np.append(ys_pred, yr_pred)

    def _sim_pop(
        self, nb_reps: int, nb_areas: int, samp_size: int, scale: np.ndarray
    ) -> np.ndarray:

        error = np.random.normal(
            loc=0, scale=(scale ** 2) * self.error_std, size=(nb_reps, samp_size)
        )
        re = np.random.normal(loc=0, scale=self.re_std, size=(nb_reps, nb_areas))
        y = np.matmul(X, self.fixed_effects) + np.repeat(re, Nd, axis=1) + error

        return y

    def fit(
        self,
        y: Array,
        X: Array,
        area: Array,
        samp_weight: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
    ) -> None:

        area = formats.numpy_array(area)
        y = formats.numpy_array(y)
        X = formats.numpy_array(X)
        if intercept and isinstance(X, np.ndarray):
            X = np.insert(X, 0, 1, axis=1)

        if samp_weight is not None and isinstance(samp_weight, pd.DataFrame):
            samp_weight = formats.numpy_array(samp_weight)

        if isinstance(scale, (float, int)):
            scale = np.ones(y.shape[0]) * scale
        else:
            scale = formats.numpy_array(scale)

        self.scale = self._sumby(area, scale)

        reml = True if self.method == "REML" else False
        basic_model = sm.MixedLM(y, X, area)
        basic_fit = basic_model.fit(reml=reml, full_output=True)
        self.area_s = np.unique(formats.numpy_array(area))

        self.error_std = basic_fit.scale ** 0.5
        self.fixed_effects = basic_fit.fe_params
        # self.random_effects = np.array(list(basic_fit.random_effects.values()))

        self.fe_std = basic_fit.bse_fe
        self.re_std = float(basic_fit.cov_re) ** 0.5
        self.re_std_cov = basic_fit.bse_re
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

        self.ybar_s, self.xbar_s, self.gamma, self.samp_size = self._area_stats(
            y, X, area, samp_weight, scale
        )

        # samp_weight = np.ones(y.size)
        if samp_weight is not None:
            beta_w = self._beta(y, X, area, samp_weight, scale)
            # print(beta_w)

        self.fitted = True

    def predict(
        self,
        X: Array,
        Xmean: Array,
        area: Array,
        pop_size: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
    ) -> None:

        if not self.fitted:
            raise ("The model must be fitted first with .fit() before running the prediction.")

        if isinstance(scale, (float, int)):
            scale = np.ones(y.shape[0]) * scale
        else:
            scale = formats.numpy_array(scale)

        area = formats.numpy_array(area)
        self.area_p = np.unique(formats.numpy_array(area))

        X = formats.numpy_array(X)
        Xmean = formats.numpy_array(Xmean)
        self.Xbar_p = Xmean
        if intercept:
            X = np.insert(X, 0, 1, axis=1)
            Xmean = np.insert(Xmean, 0, 1, axis=1)
        if pop_size is not None:
            pop_size = formats.numpy_array(pop_size)

        self.random_effects = self.gamma * (
            self.ybar_s - np.matmul(self.xbar_s, self.fixed_effects)
        )

        ps = np.isin(area, self.area_s)
        area_ps = area[ps]
        areas = np.unique(area)
        areas_ps = np.unique(area_ps)

        X_ps = X[ps]
        ps_area = np.isin(areas, areas_ps)
        Xmean_ps = Xmean[ps_area]
        Xmean_pr = Xmean[~ps_area]
        xbar_ps = self.xbar_s[ps_area]
        scale_ps = self.scale[ps_area]
        samp_size_ps = self.samp_size[ps_area]
        gamma_ps = self.gamma[ps_area]

        samp_rate_ps = samp_size_ps / pop_size[ps_area]

        self.y_predicted = self._predict(
            pop_size, Xmean_ps, Xmean_pr, xbar_ps, gamma_ps, samp_rate_ps, ps, ps_area,
        )

        A_inv = np.linalg.inv(self._A_matrix(area_ps, X_ps))
        g1 = self._g1(gamma_ps, scale_ps)
        g2 = self._g2(areas_ps, xbar_ps, Xmean_ps, gamma_ps, A_inv)
        g3 = self._g3(self.error_std ** 2, self.re_std ** 2, scale_ps, samp_size_ps)

        mse_ps = g1 + g2 + 2 * g3

        print(f"The MSE estimator is:\n {mse_ps}\n")


class RobustUnitLevel:
    """implement the robust unit level model"""

    pass
