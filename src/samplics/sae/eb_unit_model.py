from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import math

import statsmodels.api as sm

from scipy.stats import boxcox, norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum
from samplics.sae.core_sae_functions import fixed_coefficients, iterative_fisher_scoring


class EblupUnitLevel:
    """implements BHL model
    """

    def __init__(
        self, method: str = "REML",
    ):
        # Setting
        self.method: str = method.upper()
        self.boxcox: Dict[str, Optional(float)] = {"lambda": None}

        # Sample data
        self.y_s: np.ndarray = np.array([])
        self.X_s: np.ndarray = np.array([])
        self.area_s: np.ndarray = np.array([])
        self.areas_s: np.ndarray = np.array([])
        self.samp_size: np.ndarray = np.array([])
        self.ybar_s: np.ndarray = np.array([])
        self.xbar_s: np.ndarray = np.array([])

        # Fitted data
        self.fitted: boolean = False
        self.fixed_effects: np.ndarray = np.array([])
        self.fe_std: np.ndarray = np.array([])
        self.random_effects: np.ndarray = np.array([])
        self.re_std: float = 0
        self.re_std_cov: float = 0
        self.error_std: float = 0
        self.convergence: Dict[str, Union[float, int, bool]] = {}
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC
        self.gamma: np.ndarray = np.array([])
        self.a_factor: np.ndarray = np.array([])

        # Predict(ion/ed) data
        self.areas_p: np.ndarray = np.array([])
        self.pop_size: np.ndarray = np.array([])
        self.Xbar_p: np.ndarray = np.array([])
        self.number_reps: int = 0
        self.area_est: Dict[Any, float] = {}
        self.area_mse: Dict[Any, float] = {}

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
        for d in np.unique(area):
            aread = area == d
            w_d = weight[aread]
            y_d = y[aread]
            X_d = X[aread]
            Xw_d = Xw[aread]
            Xw_d_bar = np.sum(Xw_d, axis=0) / np.sum(w_d)
            resid_d_w = X_d - Xw_d_bar * self.gamma[d]
            beta1 = beta1 + np.matmul(np.transpose(Xw_d), resid_d_w)
            beta2 = beta2 + np.sum(resid_d_w * y_d[:, None] * w_d[:, None], axis=0)

        beta = np.matmul(np.linalg.inv(beta1), beta2)

        return beta

    @staticmethod
    def _area_stats(
        y: np.ndarray,
        X: np.ndarray,
        area: np.ndarray,
        error_std: float,
        re_std: float,
        samp_weight: Optional[np.ndarray],
        scale: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if samp_weight is None:
            weight = np.ones(y.size)

        a_factor = 1 / (scale ** 2)

        areas = np.unique(area)
        y_mean = np.zeros(areas.size)
        X_mean = np.zeros((areas.size, X.shape[1]))
        gamma = np.zeros(areas.size)
        samp_size = np.zeros(areas.size)
        for k, d in enumerate(areas):
            sample_d = area == d
            a_factor_d = a_factor[sample_d]
            weight_d = weight[sample_d]
            aw_factor_d = weight_d * a_factor_d
            yw_d = y[sample_d] * a_factor_d
            y_mean[k] = np.sum(yw_d) / np.sum(aw_factor_d)
            Xw_d = X[sample_d, :] * aw_factor_d[:, None]
            X_mean[k, :] = np.sum(Xw_d, axis=0) / np.sum(aw_factor_d)
            if samp_weight is None:
                delta_d = 1 / np.sum(a_factor_d)
            else:
                delta_d = np.sum((weight_d / np.sum(weight_d)) ** 2)
            gamma[k] = (re_std ** 2) / (re_std ** 2 + (error_std ** 2) * delta_d)
            samp_size[k] = np.sum(sample_d)

        return y_mean, X_mean, gamma, samp_size.astype(int)

    def _mse(
        self,
        areas: np.ndarray,
        Xs_mean: np.ndarray,
        Xp_mean: np.ndarray,
        gamma: np.ndarray,
        samp_size: np.ndarray,
        scale: np.ndarray,
        A_inv: np.ndarray,
    ) -> np.ndarray:

        sigma2e = self.error_std ** 2
        sigma2u = self.re_std ** 2

        g1 = gamma * sigma2e / scale

        xbar_diff = Xp_mean - gamma[:, None] * Xs_mean
        g2_matrix = np.matmul(np.matmul(xbar_diff, A_inv), np.transpose(xbar_diff))
        g2 = np.diag(g2_matrix)

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

        return g1 + g2 + 2 * g3

    @staticmethod
    def _sumby(group, y):  # Could use pd.grouby().sum(), may scale better

        groups = np.unique(group)
        sums = np.zeros(groups.size)
        for k, gr in enumerate(groups):
            sums[k] = np.sum(y[group == gr])

        return sums

    def _split_data(
        self, area: np.ndarray, X: np.ndarray, Xmean: np.ndarray, samp_weight: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        ps = np.isin(area, self.area_s)
        area_ps = area[ps]
        areas = np.unique(area)
        areas_ps = np.unique(area_ps)

        X_ps = X[ps]
        ps_area = np.isin(areas, areas_ps)
        if Xmean is not None:
            Xmean_ps = Xmean[ps_area]
            Xmean_pr = Xmean[~ps_area]
        else:
            Xmean_ps = Xmean_pr = None
        xbar_ps = self.xbar_s[ps_area]
        a_factor_ps = self.a_factor[ps_area]
        samp_size_ps = self.samp_size[ps_area]
        gamma_ps = self.gamma[ps_area]
        samp_weight_ps = samp_weight[ps] if samp_weight is not None else None

        return (
            ps,
            ps_area,
            X_ps,
            area_ps,
            areas_ps,
            Xmean_ps,
            Xmean_pr,
            xbar_ps,
            a_factor_ps,
            samp_size_ps,
            gamma_ps,
            samp_weight_ps,
        )

    def fit(
        self,
        y: Array,
        X: Array,
        area: Array,
        samp_weight: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
        tol: float = 1e-4,
        maxiter: int = 200,
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

        self.y_s = y
        self.X_s = X
        self.area_s = area
        self.areas_s = np.unique(area)

        self.a_factor = self._sumby(area, scale)

        reml = True if self.method == "REML" else False
        beta_ols = sm.OLS(y, X).fit().params
        resid_ols = y - np.matmul(X, beta_ols)
        re_ols = self._sumby(area, resid_ols) / self._sumby(area, np.ones(area.size))

        basic_model = sm.MixedLM(y, X, area)
        basic_fit = basic_model.fit(
            start_params=np.append(beta_ols, np.std(re_ols) ** 2),
            reml=reml,
            full_output=True,
            tol=tol,
            maxiter=maxiter,
        )

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

        self.ybar_s, self.xbar_s, gamma, samp_size = self._area_stats(
            y, X, area, self.error_std, self.re_std, samp_weight, scale
        )
        self.random_effects = gamma * (self.ybar_s - np.matmul(self.xbar_s, self.fixed_effects))
        self.gamma = gamma
        self.samp_size = samp_size

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
        samp_weight: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
    ) -> None:

        if not self.fitted:
            raise ("The model must be fitted first with .fit() before running the prediction.")

        if samp_weight is not None and isinstance(samp_weight, pd.DataFrame):
            samp_weight = formats.numpy_array(samp_weight)

        if isinstance(scale, (float, int)):
            scale = np.ones(X.shape[0]) * scale
        else:
            scale = formats.numpy_array(scale)

        area = formats.numpy_array(area)
        X = formats.numpy_array(X)
        Xmean = formats.numpy_array(Xmean)
        self.Xbar_p = dict(zip(area, Xmean))

        if intercept:
            X = np.insert(X, 0, 1, axis=1)
            Xmean = np.insert(Xmean, 0, 1, axis=1)
        if pop_size is not None:
            pop_size = formats.numpy_array(pop_size)

        (
            ps,
            ps_area,
            X_ps,
            area_ps,
            areas_ps,
            Xmean_ps,
            Xmean_pr,
            xbar_ps,
            a_factor_ps,
            samp_size_ps,
            gamma_ps,
            samp_weight_ps,
        ) = self._split_data(area, X, Xmean, samp_weight)

        samp_rate_ps = samp_size_ps / pop_size[ps_area]

        if pop_size is None:
            eta_pred = np.matmul(Xmean_ps, self.fixed_effects) + gamma_ps
        else:
            eta_pred = np.matmul(Xmean_ps, self.fixed_effects) + (
                samp_rate_ps + (1 - samp_rate_ps) * gamma_ps
            ) * (self.ybar_s[ps_area] - np.matmul(xbar_ps, self.fixed_effects))
        if np.sum(~ps) > 0:
            eta_r_pred = np.matmul(Xmean_pr, self.fixed_effects)
            eta_pred = np.append(eta_pred, eta_r_pred)
        self.area_est = dict(zip(areas_ps, eta_pred))

        A_ps = np.diag(np.zeros(X_ps.shape[1]))
        for d in areas_ps:
            n_ps_d = np.sum(area_ps == d)
            X_ps_d = X_ps[area_ps == d]
            V_ps_d = (self.error_std ** 2) * np.diag(np.ones(n_ps_d)) + (
                self.re_std ** 2
            ) * np.ones([n_ps_d, n_ps_d])
            A_ps = A_ps + np.matmul(np.matmul(np.transpose(X_ps_d), np.linalg.inv(V_ps_d)), X_ps_d)
        mse_ps = self._mse(
            areas_ps, xbar_ps, Xmean_ps, gamma_ps, samp_size_ps, a_factor_ps, np.linalg.inv(A_ps)
        )
        self.area_mse = dict(zip(areas_ps, mse_ps))

        # TODO: add non-sampled areas prediction

    def bootstrap_mse(
        self,
        number_reps: int,
        X: np.ndarray,
        Xmean: Array,
        area: np.ndarray,
        samp_weight: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
        tol: float = 1e-4,
        maxiter: int = 200,
    ) -> np.ndarray:

        X = formats.numpy_array(X)
        Xmean = formats.numpy_array(Xmean)
        area = formats.numpy_array(area)

        if intercept:
            X = np.insert(X, 0, 1, axis=1)
            Xmean = np.insert(Xmean, 0, 1, axis=1)

        if samp_weight is not None and isinstance(samp_weight, pd.DataFrame):
            samp_weight = formats.numpy_array(samp_weight)

        if isinstance(scale, (float, int)):
            scale = np.ones(X.shape[0]) * scale
        else:
            scale = formats.numpy_array(scale)

        self.a_factor = self._sumby(area, scale)

        (
            ps,
            ps_area,
            X_ps,
            area_ps,
            areas_ps,
            Xmean_ps,
            Xmean_pr,
            xbar_ps,
            a_factor_ps,
            samp_size_ps,
            gamma_ps,
            samp_weight_ps,
        ) = self._split_data(area, X, Xmean, samp_weight)

        X_ps_sorted = X_ps[np.argsort(area_ps)]
        scale_ps_ordered = scale[ps]
        area_ps_sorted = area_ps[np.argsort(area_ps)]
        if np.min(scale_ps_ordered) != np.max(scale_ps_ordered):
            scale_ps_ordered = scale_ps_ordered[np.argsort(area_ps)]

        nb_areas = len(np.unique(area_ps))

        error = np.abs(scale_ps_ordered) * np.random.normal(
            loc=0, scale=self.error_std, size=(number_reps, area_ps.size)
        )
        re = np.random.normal(loc=0, scale=self.re_std, size=(number_reps, nb_areas))
        mu = np.matmul(X_ps_sorted, self.fixed_effects)
        y_ps_boot = (
            np.repeat(mu[None, :], number_reps, axis=0)
            + np.repeat(re, samp_size_ps, axis=1)
            + error
        )

        bar_length = min(50, number_reps)
        steps = np.linspace(0, number_reps, bar_length).astype(int)
        i = 0

        reml = True if self.method == "REML" else False
        boot_mse = np.zeros((number_reps, nb_areas))
        print(f"Running the {number_reps} bootstrap iterations")
        for k in range(y_ps_boot.shape[0]):
            boot_model = sm.MixedLM(y_ps_boot[k, :], X_ps_sorted, area_ps)
            boot_fit = boot_model.fit(
                start_params=np.append(self.fixed_effects, self.re_std ** 2),
                reml=reml,
                tol=tol,
                maxiter=maxiter,
            )
            boot_fe = boot_fit.fe_params
            boot_error_std = boot_fit.scale ** 0.5
            boot_re_std = float(boot_fit.cov_re) ** 0.5
            boot_ybar_s, boot_xbar_s, boot_gamma, _ = self._area_stats(
                y_ps_boot[k, :],
                X_ps_sorted,
                area_ps,
                boot_error_std,
                boot_re_std,
                samp_weight_ps,
                scale_ps_ordered,
            )
            boot_re = boot_gamma * (boot_ybar_s - np.matmul(boot_xbar_s, boot_fe))
            boot_mu = np.matmul(Xmean_ps, self.fixed_effects) + re[k, :]
            boot_mu_h = np.matmul(Xmean_ps, boot_fe) + boot_re
            boot_mse[k, :] = (boot_mu_h - boot_mu) ** 2

            if k in steps:
                i += 1
                print(
                    f"\r[%-{bar_length-1}s] %d%%" % ("=" * i, 2 + (100 / bar_length) * i), end="",
                )
        print("\n")

        return np.mean(boot_mse, axis=0)


class EbUnitLevel:
    """implements the unit level model"""

    def __init__(
        self, method: str = "REML", indicator=None, boxcox: Optional[float] = None,
    ):

        # Setting
        self.method: str = method.upper()
        self.number_samples: Optional[int] = None
        if boxcox is None:
            self.boxcox = {"lambda": None}
        else:
            self.boxcox = {"lambda": boxcox}

        # Sample data
        self.y_s: np.ndarray = np.array([])
        self.X_s: np.ndarray = np.array([])
        self.area_s: np.ndarray = np.array([])
        self.areas_s: np.ndarray = np.array([])
        self.samp_size: np.ndarray = np.array([])
        self.ybar_s: np.ndarray = np.array([])
        self.xbar_s: np.ndarray = np.array([])

        # Fitted data
        self.fitted: boolean = False
        self.fixed_effects: np.ndarray = np.array([])
        self.fe_std: np.ndarray = np.array([])
        self.random_effects: np.ndarray = np.array([])
        self.re_std: float = 0
        self.re_std_cov: float = 0
        self.error_std: float = 0
        self.convergence: Dict[str, Union[float, int, bool]] = {}
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC
        self.gamma: np.ndarray = np.array([])
        self.a_factor: np.ndarray = np.array([])

        # Predict(ion/ed) data
        self.areas_p: np.ndarray = np.array([])
        self.pop_size: np.ndarray = np.array([])
        self.Xbar_p: np.ndarray = np.array([])
        self.number_reps: int = 0
        self.area_est: Dict[Any, float] = {}
        self.area_mse: Dict[Any, float] = {}

    def fit(
        self,
        y: Array,
        X: Array,
        area: Array,
        samp_weight: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
        tol: float = 1e-4,
        maxiter: int = 200,
    ) -> None:

        eblupUL = EblupUnitLevel()
        eblupUL.fit(y, X, area, samp_weight, scale, intercept, tol, maxiter)

        self.y_s = y
        self.X_s = X
        self.area_s = area

        self.areas_s = eblupUL.areas_s
        self.a_factor = eblupUL.a_factor
        self.error_std = eblupUL.error_std

        self.fixed_effects = eblupUL.fixed_effects
        self.fe_std = eblupUL.fe_std
        self.re_std = eblupUL.re_std
        self.re_std_cov = eblupUL.re_std_cov
        self.convergence = eblupUL.convergence
        self.goodness = eblupUL.goodness

        self.ybar_s = eblupUL.ybar_s
        self.xbar_s = eblupUL.xbar_s
        self.gamma = eblupUL.gamma
        self.samp_size = eblupUL.samp_size

        self.fitted = eblupUL.fitted

    @staticmethod
    def _predict_indicator(
        number_samples,
        y_s,
        X_s,
        area_s,
        X_r,
        area_r,
        areas_r,
        fixed_effects,
        gamma,
        sigma2e,
        sigma2u,
        scale,
        max_array_length,
        indicator,
        *args,
    ):
        nb_areas_r = len(areas_r)
        mu_r = X_r @ fixed_effects

        k = 0
        bar_length = min(50, nb_areas_r)
        steps = np.linspace(1, nb_areas_r - 1, bar_length).astype(int)

        print(f"Generating the {number_samples} replicates samples\n")
        eta = np.zeros((number_samples, nb_areas_r)) * np.nan
        for i, d in enumerate(areas_r):
            # print(d)
            oos = area_r == d
            mu_dr = mu_r[oos]
            gamma_dr = gamma[i]
            scale_dr = scale[oos]
            N_dr = np.sum(oos)
            cycle_size = max(int(max_array_length // N_dr), 1)
            number_cycles = int(number_samples // cycle_size)
            last_cycle_size = number_samples % cycle_size

            # print(
            #     f"Calculation for the {d}th domain with a total of {number_cycles+1} batch(es)\n"
            # )
            for j in range(number_cycles + 1):
                if j == number_cycles:
                    cycle_size = last_cycle_size
                # if cycle_size > 0:
                #     print(f"{k+1}th batch with {cycle_size} samples (domain {d})")
                re_effects = np.random.normal(
                    scale=(sigma2u * (1 - gamma_dr)) ** 0.5, size=cycle_size
                )
                errors = np.random.normal(
                    scale=scale_dr * (sigma2e ** 0.5), size=(cycle_size, N_dr)
                )
                y_dr_j = mu_dr[None, :] + re_effects[:, None] + errors
                if j == 0:
                    y_dr = y_dr_j
                else:
                    y_dr = np.append(y_dr, y_dr_j, axis=0)

            if i in steps:
                k += 1
                print(
                    f"\r[%-{bar_length}s] %d%%" % ("=" * k, 2 * k), end="",
                )

            y_d = np.append(y_dr, np.tile(y_s[area_s == d], [number_samples, 1]), axis=1)
            eta[:, i] = np.apply_along_axis(indicator, axis=1, arr=y_d, *args)  # *)

        print("\n")

        return np.mean(eta, axis=0)

    def predict(
        self,
        indicator,
        number_samples,
        X,
        area,
        samp_weight=None,
        scale=1,
        intercept=True,
        max_array_length=1e6,
        *args,
    ):

        if not self.fitted:
            raise ("The model must be fitted first with .fit() before running the prediction.")

        self.number_samples = int(number_samples)

        if samp_weight is not None and isinstance(samp_weight, pd.DataFrame):
            samp_weight = formats.numpy_array(samp_weight)

        if isinstance(scale, (float, int)):
            scale = np.ones(X.shape[0]) * scale
        else:
            scale = formats.numpy_array(scale)
        area = formats.numpy_array(area)
        self.areas_p = np.unique(area)
        X = formats.numpy_array(X)
        if intercept:
            X = np.insert(X, 0, 1, axis=1)

        # (
        #     ps,
        #     ps_area,
        #     X_ps,
        #     area_ps,
        #     areas_ps,
        #     _,
        #     _,
        #     xbar_ps,
        #     a_factor_ps,
        #     samp_size_ps,
        #     gamma_ps,
        #     samp_weight_ps,
        # ) = EblupUnitLevel._split_data(area, X, None, samp_weight)

        area_est = self._predict_indicator(
            self.number_samples,
            self.y_s,
            self.X_s,
            self.area_s,
            X,
            area,
            self.areas_p,
            self.fixed_effects,
            self.gamma,
            self.error_std ** 2,
            self.re_std ** 2,
            scale,
            max_array_length,
            indicator,
            *args,
        )

        self.area_est = dict(zip(self.areas_s, area_est))

    def bootstrap_mse(
        self,
        number_reps: int,
        indicator,
        X: np.ndarray,
        area: np.ndarray,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
        tol: float = 1e-4,
        maxiter: int = 200,
        max_array_length=100e6,
    ) -> np.ndarray:

        X_p = formats.numpy_array(X)
        area_p = formats.numpy_array(area)
        areas_p = np.unique(area_p)

        if intercept:
            X_p = np.insert(X_p, 0, 1, axis=1)
            X_s = np.insert(self.X_s, 0, 1, axis=1)
        else:
            X_s = self.X_s

        if isinstance(scale, (float, int)):
            scale = np.ones(X_p.shape[0]) * scale
        else:
            scale = formats.numpy_array(scale)

        ps = np.isin(area_p, self.areas_s)
        area_ps, area_pr = area_p[ps], area_p[~ps]

        areas, Nd = np.unique(
            np.append(area_p, self.area_s[np.isin(self.area_s, areas_p)]), return_counts=True
        )
        X = np.append(X_p, X_s[np.isin(self.area_s, areas_p)], axis=0)
        # X_ps, X_pr = X[ps], X[~ps]
        # mu_ps = X_ps @ self.fixed_effects
        # mu_pr = X_pr @ self.fixed_effects
        # mu_p = np.append(mu_ps, mu_pr, axis=0)
        for b in range(number_reps):
            re_effects = np.random.normal(scale=self.re_std * (1 - self.gamma) ** 0.5)
            errors = np.random.normal(scale=scale * self.error_std)
            yboot = X @ self.fixed_effects + np.repeat(re_effects, Nd) + errors

            print(yboot)
        exit()

        for d in areas:
            aread = area == d
            X_d = X[aread]
            X_ds = X_s[self.area_s == d]
            scale_d = scale[aread]
            N_dr = np.sum(aread).astype(int)
            d_sampled = self.areas_s == d
            n_ds = int(self.samp_size[d_sampled]) if d in self.areas_s else 0
            N_d = N_dr + n_ds

            y_d = (
                X_d @ self.fixed_effects
                + np.random.normal(scale=self.re_std ** (1 / 2))
                + np.random.normal(scale=scale_d * (self.error_std ** (1 / 2)))
            )

            y_d = np.append(y_d, self.y_s)
            X_d = np.append(X_d, X_ds, axis=0)
            area_d = np.append(area[aread], self.area_s)

            param_d = indicator(y_d)
            sample_d = np.random.choice(N_d, size=n_ds, replace=False)

            yboot_ds = y_d[sample_d]
            Xboot_ds = X_d[sample_d]
            areaboot_ds = area_d[sample_d]

            reml = True if self.method == "REML" else False
            beta_ols = sm.OLS(yboot_ds, Xboot_ds).fit().params
            resid_ols = yboot_ds - np.matmul(Xboot_ds, beta_ols)
            re_ols = EblupUnitLevel._sumby(areaboot_ds, resid_ols) / EblupUnitLevel._sumby(
                areaboot_ds, np.ones(areaboot_ds.size)
            )

            boot_model = sm.MixedLM(yboot_ds, Xboot_ds, areaboot_ds)
            boot_fit = boot_model.fit(
                start_params=np.append(beta_ols, np.std(re_ols) ** 2),
                reml=reml,
                full_output=True,
                tol=tol,
                maxiter=maxiter,
            )

            gamma_d = float(boot_fit.cov_re) / (
                float(boot_fit.cov_re) + boot_fit.scale * (1 / np.sum(scale_d ** 2))
            )

            area_est = self._predict_indicator(
                self.number_samples,
                yboot_ds,
                Xboot_ds,
                areaboot_ds,
                X_d,
                np.unique(area_d),
                self.areas_p,
                boot_fit.fe_params,
                gamma_d,
                boot_fit.scale,
                float(boot_fit.cov_re),
                scale_d,
                max_array_length,
                indicator,
                *args,
            )

        return yboot_ds


class EllUnitLevel:
    """implement the ELL unit level model"""

    pass


class RobustUnitLevel:
    """implement the robust unit level model"""

    pass
