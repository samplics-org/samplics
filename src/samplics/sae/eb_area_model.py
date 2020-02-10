from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import math

import statsmodels.api as sm

from scipy.stats import norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


class AreaModel:
    "Implement the area level model"

    # Other option for method are spatial (SP), temporal (TP), spatial-temporal (ST)
    def __init__(self, model: str = "FH") -> None:
        self.model: str = model.upper()
        self.fe_coef: np.ndarray = np.array([])
        self.fe_cov: np.ndarray = np.array([])
        self.re_coef: np.ndarray = np.array([])
        self.re_cov: np.ndarray = np.array([])
        self.method: str = "REML"
        self.convergence: Dict[str, Union[float, int, bool]] = {}
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC
        self.point_est: Dict[Any, float] = {}
        self.mse: Dict[Any, float] = {}
        self.mse_as1: Dict[Any, float] = {}
        self.mse_as2: Dict[Any, float] = {}
        self.mse_terms: Dict[str, Dict[Any, float]] = {}

    def __str__(self) -> str:

        estimation = pd.DataFrame()
        estimation["area"] = self.area
        estimation["estimate"] = self.point_est
        estimation["mse"] = self.mse

        fit = pd.DataFrame()
        fit["beta_coef"] = self.fe_coef
        fit["beta_stderr"] = np.diag(self.fe_cov) ** (1 / 2)

        return f"""\n\n{self.model} Area Model - Best predictor,\n\nConvergence status: {self.convergence['achieved']}\nNumber of iterations: {self.convergence['iterations']}\nPrecision: {self.convergence['precision']}\n\nGoodness of fit: {self.goodness}\n\nEstimation:\n{estimation}\n\nFixed effect:\n{fit}\n\nRandom effect variance:\n{self.re_coef}\n\n"""

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _fixed_coefs(
        area: np.ndarray,
        yhat: np.ndarray,
        X: np.ndarray,
        sigma2_e: np.ndarray,
        sigma2_v: float,
        b_const: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """[summary]
        
        Arguments:
            area {np.ndarray} -- [description]
            yhat {np.ndarray} -- [description]
            X {np.ndarray} -- [description]
            sigma2_e {np.ndarray} -- [description]
            sigma2_v {float} -- [description]
            b_const {np.ndarray} -- [description]
        
        Returns:
            Tuple[np.ndarray, np.ndarray] -- [description]
        """

        V = np.diag(sigma2_v * (b_const ** 2) + sigma2_e)
        V_inv = np.linalg.inv(V)
        x_v_X_inv = np.linalg.inv(np.matmul(np.matmul(np.transpose(X), V_inv), X))
        x_v_x_inv_x = np.matmul(np.matmul(x_v_X_inv, np.transpose(X)), V_inv)
        beta_hat = np.matmul(x_v_x_inv_x, yhat)
        beta_cov = np.matmul(np.matmul(np.transpose(X), V_inv), X)

        return beta_hat.ravel(), np.linalg.inv(beta_cov)

    def _likelihood(self, y: np.ndarray, X: np.ndarray, beta: np.ndarray, V: np.ndarray) -> float:

        m = y.size
        const = m * np.log(2 * np.pi)
        ll_term1 = np.log(np.linalg.det(V))
        V_inv = np.linalg.inv(V)
        resid_term = y - np.dot(X, beta)
        if self.method in ("ML", "FH"):  # Whta is likelihood for FH
            resid_var = np.dot(np.transpose(resid_term), V_inv)
            ll_term2 = np.dot(resid_var, resid_term)
            loglike = -0.5 * (const + ll_term1 + ll_term2)
        elif self.method == "REML":
            xT_vinv_x = np.dot(np.dot(np.transpose(X), V_inv), X)
            ll_term2 = np.log(np.linalg.det(xT_vinv_x))
            ll_term3 = np.dot(np.dot(y, V_inv), resid_term)
            loglike = -0.5 * (const + ll_term1 + ll_term2 + ll_term3)
        else:
            raise AssertionError("A fitting method must be specified.")

        return float(loglike)

    def _derivatives_sigma(
        self,
        area: np.ndarray,
        yhat: np.ndarray,
        X: np.ndarray,
        sigma2_e: np.ndarray,
        sigma2_v: np.ndarray,
        b_const: np.ndarray,
    ) -> Tuple[float, float]:

        if self.method == "ML":
            beta, beta_cov = self._fixed_coefs(
                area=area, yhat=yhat, X=X, sigma2_e=sigma2_e, sigma2_v=sigma2_v, b_const=b_const
            )
            deriv_sigma = 0.0
            info_sigma = 0.0
            for d in area:
                b_d = b_const[area == d]
                phi_d = sigma2_e[area == d]
                X_d = X[area == d, :]
                yhat_d = yhat[area == d]
                mu_d = np.matmul(X_d, beta)
                resid_d = yhat_d - mu_d
                sigma2_d = sigma2_v * (b_d ** 2) + phi_d
                term1 = float(b_d ** 2 / sigma2_d)
                term2 = float(((b_d ** 2) * (resid_d ** 2)) / (sigma2_d ** 2))
                deriv_sigma += -0.5 * (term1 - term2)
                info_sigma += 0.5 * (term1 ** 2)
        elif self.method == "REML":
            B = np.diag(b_const ** 2)
            v_i = sigma2_e + sigma2_v * (b_const ** 2)
            V = np.diag(v_i)
            v_inv = np.linalg.inv(V)
            x_vinv_x = np.matmul(np.matmul(np.transpose(X), v_inv), X)
            x_xvinvx_x = np.matmul(np.matmul(X, np.linalg.inv(x_vinv_x)), np.transpose(X))
            P = v_inv - np.matmul(np.matmul(v_inv, x_xvinvx_x), v_inv)
            P_B = np.matmul(P, B)
            P_B_P = np.matmul(P_B, P)
            term1 = np.trace(P_B)
            term2 = np.matmul(np.matmul(np.transpose(yhat), P_B_P), yhat)
            deriv_sigma = -0.5 * (term1 - term2)
            info_sigma = 0.5 * np.trace(np.matmul(P_B_P, B))
        elif self.method == "FH":  # Fay-Herriot approximation
            beta, beta_cov = self._fixed_coefs(
                area=area, yhat=yhat, X=X, sigma2_e=sigma2_e, sigma2_v=sigma2_v, b_const=b_const
            )
            deriv_sigma = 0.0
            info_sigma = 0.0
            for d in area:
                b_d = b_const[area == d]
                phi_d = sigma2_e[area == d]
                X_d = X[area == d, :]
                yhat_d = yhat[area == d]
                mu_d = np.dot(X_d, beta)
                resid_d = yhat_d - mu_d
                sigma2_d = sigma2_v * (b_d ** 2) + phi_d
                deriv_sigma += float((resid_d ** 2) / sigma2_d)
                info_sigma += -float(((b_d ** 2) * (resid_d ** 2)) / (sigma2_d ** 2))
            m = yhat.size
            p = X.shape[1]
            deriv_sigma = m - p - deriv_sigma

        return float(deriv_sigma), float(info_sigma)

    def _iterative_methods(
        self,
        area: np.ndarray,
        yhat: np.ndarray,
        X: np.ndarray,
        sigma2_e: np.ndarray,
        b_const: np.ndarray,
        sigma2_v_start: float,
        maxiter: int,
        abstol: float,
        reltol: float,
    ) -> Tuple[float, float, int, float, bool]:  # May not need variance
        """ Fisher-scroring algorithm for estimation of variance component"""

        iterations = 0

        tolerance = abstol + 1.0
        tol = 0.9 * tolerance
        sigma2_v = sigma2_v_start
        while tolerance > tol:
            sigma2_v_previous = sigma2_v
            deriv_sigma, info_sigma = self._derivatives_sigma(
                area=area, yhat=yhat, X=X, sigma2_e=sigma2_e, sigma2_v=sigma2_v, b_const=b_const,
            )

            sigma2_v += deriv_sigma / info_sigma
            sigma2_v_cov = 1 / info_sigma

            tolerance = abs(sigma2_v - sigma2_v_previous)
            tol = max(abstol, reltol * abs(sigma2_v))
            convergence = tolerance <= tol

            if iterations == maxiter:
                break
            else:
                iterations += 1

        sigma2_v = float(max(sigma2_v, 0))

        return sigma2_v, sigma2_v_cov, iterations, tolerance, convergence

    def _eb_estimates(
        self,
        yhat: np.ndarray,
        X: np.ndarray,
        beta: np.ndarray,
        area: np.ndarray,
        sigma2_e: np.ndarray,
        sigma2_v: float,
        sigma2_v_cov: float,
        b_const: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.dtype,
    ]:

        m = yhat.size
        v_i = sigma2_e + sigma2_v * (b_const ** 2)
        V = np.diag(v_i)
        V_inv = np.diag(1 / v_i)
        mu = np.dot(X, beta)
        resid = yhat - mu
        G = np.diag(np.ones(m) * sigma2_v)

        Z = np.diag(b_const)

        b = np.matmul(np.matmul(G, np.transpose(Z)), V_inv)
        d = np.transpose(X - np.matmul(np.transpose(b), X))
        x_vinv_x = np.matmul(np.matmul(np.transpose(X), V_inv), X)
        g2_term = np.linalg.inv(x_vinv_x)

        b_term_ml1 = np.linalg.inv(x_vinv_x)
        b_term_ml2_diag = (b_const ** 2) / (v_i ** 2)
        b_term_ml2 = np.matmul(np.matmul(np.transpose(X), np.diag(b_term_ml2_diag)), X)
        b_term_ml = np.trace(np.matmul(b_term_ml1, b_term_ml2))

        estimates = np.array(yhat) * np.nan
        g1 = np.array(yhat) * np.nan
        g2 = np.array(yhat) * np.nan
        g3 = np.array(yhat) * np.nan
        g3_star = np.array(yhat) * np.nan

        g1_partial = np.array(yhat) * np.nan

        sum_inv_vi2 = np.sum(1 / (v_i ** 2))
        if self.method == "REML":
            b_sigma2_v = 0
            g3_scale = 2 / sum_inv_vi2
        elif self.method == "ML":
            b_sigma2_v = -(1 / 2 * sigma2_v_cov) * b_term_ml
            g3_scale = 2 / sum_inv_vi2
        elif self.method == "FH":
            sum_vi = np.sum((1 / v_i))
            b_sigma2_v = 2 * (m * sum_inv_vi2 - sum_vi ** 2) / (sum_vi ** 3)
            g3_scale = 2 * m / sum_vi ** 2

        for d in area:
            b_d = b_const[area == d]
            phi_d = sigma2_e[area == d]
            X_d = X[area == d]
            yhat_d = yhat[area == d]
            mu_d = np.matmul(X_d, beta)
            resid_d = yhat_d - mu_d
            variance_d = sigma2_v * (b_d ** 2) + phi_d
            gamma_d = sigma2_v * (b_d ** 2) / variance_d
            estimates[area == d] = gamma_d * yhat_d + (1 - gamma_d) * mu_d
            g1[area == d] = gamma_d * phi_d
            g2_term_d = np.matmul(np.matmul(X_d, g2_term), np.transpose(X_d))
            g2[area == d] = ((1 - gamma_d) ** 2) * float(g2_term_d)
            g3[area == d] = ((1 - gamma_d) ** 2) * g3_scale / variance_d
            g3_star[area == d] = (g3[area == d] / variance_d) * (resid_d ** 2)
            g1_partial[area == d] = (b_d ** 2) * ((1 - gamma_d) ** 2) * b_sigma2_v

        if self.method == "REML":
            mse = g1 + g2 + 2 * g3
            mse1_area_specific = g1 + g2 + 2 * g3_star
            mse2_area_specific = g1 + g2 + g3 + g3_star
        elif self.method in ("ML", "FH"):
            mse = g1 - g1_partial + g2 + 2 * g3
            mse1_area_specific = g1 - g1_partial + g2 + 2 * g3_star
            mse2_area_specific = g1 - g1_partial + g2 + g3 + g3_star

        return (estimates, mse, mse1_area_specific, mse2_area_specific, g1, g2, g3, g3_star)

    def _fit(
        self,
        yhat: Array,
        X: Array,
        area: Array,
        sigma2_e: Array,
        sigma2_v_start: float,
        b_const: np.array,
        maxiter: int,
        abstol: float,
        reltol: float,
    ) -> None:

        (sigma2_v, sigma2_v_cov, iterations, tolerance, convergence) = self._iterative_methods(
            area=area,
            yhat=yhat,
            X=X,
            sigma2_e=sigma2_e,
            b_const=b_const,
            sigma2_v_start=sigma2_v_start,
            maxiter=maxiter,
            abstol=abstol,
            reltol=reltol,
        )

        beta, beta_cov = self._fixed_coefs(
            area=area, yhat=yhat, X=X, sigma2_e=sigma2_e, sigma2_v=sigma2_v, b_const=b_const
        )

        self.fe_coef = beta
        self.fe_cov = beta_cov

        self.re_coef = sigma2_v
        self.re_cov = sigma2_v_cov

        self.convergence["achieved"] = convergence
        self.convergence["iterations"] = iterations
        self.convergence["precision"] = tolerance

        m = yhat.size
        p = X.shape[1] + 1
        Z_b2_Z = np.ones(shape=(m, m))
        V = np.diag(sigma2_e) + sigma2_v * Z_b2_Z
        logllike = self._likelihood(yhat, X=X, beta=self.fe_coef, V=V)
        self.goodness["loglike"] = logllike
        self.goodness["AIC"] = -2 * logllike + 2 * (p + 1)
        self.goodness["BIC"] = -2 * logllike + math.log(m) * (p + 1)

    def predict(
        self,
        y_s: Array,
        X_s: Array,
        X_r: Array,
        area_s: Array,
        area_r: Array,
        sigma2_e: Array,
        sigma2_v_start: float = 0.001,
        method: str = "REML",
        b_const: Union[np.array, float, int] = 1.0,
        maxiter: int = 100,
        abstol: float = 1.0e-4,
        reltol: float = 0.0,
    ) -> Tuple[Dict[Any, float], Dict[Any, float]]:

        if method.upper() not in ("FH", "ML", "REML"):
            raise AssertionError("Parameter method must be 'FH', 'ML, or 'REML'.")
        else:
            self.method = method.upper()

        if isinstance(b_const, (int, float)):
            b_const = np.ones(area_s.size) * b_const

        area_s = formats.numpy_array(area_s)
        area_r = formats.numpy_array(area_r)
        y_s = formats.numpy_array(y_s)
        X_s = formats.numpy_array(X_s)
        X_r = formats.numpy_array(X_r)
        b_const = formats.numpy_array(b_const)

        if abstol <= 0.0 and reltol <= 0.0:
            AssertionError("At least one tolerance parameters must be positive.")
        else:
            abstol = max(abstol, 0)
            reltol = max(reltol, 0)

        self._fit(
            yhat=y_s,
            X=X_s,
            area=area_s,
            sigma2_e=sigma2_e,
            sigma2_v_start=sigma2_v_start,
            b_const=b_const,
            maxiter=maxiter,
            abstol=abstol,
            reltol=reltol,
        )

        point_est, mse, mse1, mse2, g1, g2, g3, g3_star = self._eb_estimates(
            yhat=y_s,
            X=X_s,
            area=area_s,
            beta=self.fe_coef,
            sigma2_e=sigma2_e,
            sigma2_v=self.re_coef,
            sigma2_v_cov=self.re_cov,
            b_const=b_const,
        )

        self.point_est = point_est
        self.mse = mse
        self.area = area_s

