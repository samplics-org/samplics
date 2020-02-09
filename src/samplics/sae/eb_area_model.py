from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

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
        self.iterations: int = 0
        self.convergence: bool = False
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC
        self.point_est: Dict[Any, float] = {}
        self.mse: Dict[Any, float] = {}
        self.mse_as1: Dict[Any, float] = {}
        self.mse_as2: Dict[Any, float] = {}
        self.mse_terms: Dict[str, Dict[Any, float]] = {}

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

        p = np.shape(X)[1]
        beta_term1 = np.zeros(shape=(p, p))
        beta_term2 = np.zeros(shape=(p, 1))
        for d in area:
            yhat_d = yhat[area == d]
            X_d = X[area == d]
            b_d = b_const[area == d]
            phi_d = sigma2_e[area == d]
            sigma2_d = sigma2_v * b_d ** 2 + phi_d
            beta_term1 += np.matmul(np.transpose(X_d), X_d) / sigma2_d
            beta_term2 += np.transpose(X_d) * yhat_d / sigma2_d
        beta_hat = np.matmul(np.linalg.inv(beta_term1), beta_term2)
        v_i = sigma2_e + sigma2_v * b_const ** 2
        V = np.diag(v_i)
        V_inv = np.linalg.inv(V)
        beta_cov = np.matmul(np.matmul(np.transpose(X), V_inv), X)

        return beta_hat.ravel(), beta_cov.ravel()

    def _likelihood(
        self, yhat: np.ndarray, X: np.ndarray, beta: np.ndarray, V: np.ndarray
    ) -> float:
        """[summary]
        
        Args:
            yhat (np.ndarray): [description]
            X (np.ndarray): [description]
            beta (np.ndarray): [description]
            V (np.ndarray): [description]
        
        Raises:
            AssertionError: [description]
        
        Returns:
            float: [description]
        """

        if self.method == "ML":
            ll_term1 = np.log(abs(V))
            resid_term = yhat - np.matmul(X, beta)
            V_inv = np.linalg.inv(V)
            resid_var = np.matmul(np.transpose(resid_term), V_inv)
            ll_term2 = np.matmul(resid_var, resid_term)
            loglike = -0.5 * (ll_term1 + ll_term2)
        elif self.method == "REML":
            ll_term1 = np.log(abs(V))
            v_inv = np.linalg.inv(V)
            xT_vinv_x = np.matmul(np.matmul(np.transpose(X), v_inv), X)
            ll_term2 = np.log(abs(xT_vinv_x))
            resid_term = yhat - np.matmul(X, beta)
            ll_term3 = np.log(np.matmul(np.matmul(yhat, v_inv), resid_term))
            loglike = -0.5 * (ll_term1 + ll_term2 + ll_term3)
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
                X_d = X[area == d]
                yhat_d = yhat[area == d]
                mu_d = np.matmul(X_d, beta)
                resid_d = yhat_d - mu_d
                sigma2_d = sigma2_v * b_d ** 2 + phi_d
                term1 = b_d ** 2 / sigma2_d
                term2 = (b_d * resid_d) ** 2 / sigma2_d ** 2
                deriv_sigma += -0.5 * (term1 - term2)
                info_sigma += 0.5 * term1 ** 2
        elif self.method == "REML":
            B = np.diag(b_const ** 2)
            v_i = sigma2_e + sigma2_v * b_const ** 2
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
                X_d = X[area == d]
                yhat_d = yhat[area == d]
                mu_d = np.matmul(X_d, beta)
                resid_d = yhat_d - mu_d
                sigma2_d = sigma2_v * b_d ** 2 + phi_d
                deriv_sigma += resid_d ** 2 / sigma2_d
                info_sigma += -((b_d * resid_d) ** 2) / sigma2_d ** 2
            m = np.size(area)
            p = np.shape(X)[1]
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
        maxiter: int = 100,
        tol: float = 1.0e-6,
        rtol: float = 0.0,
    ) -> Tuple[float, float, int, float, bool]:  # May not need variance
        """ Fisher-scroring algorithm for estimation of variance component"""

        iterations = 1
        if rtol < 0.0:
            AssertionError("Relative tolerance must be positive.")
        elif rtol != 0.0:
            tol = rtol

        tolerance = tol + 1.0
        sigma2_v = sigma2_v_start
        while tolerance > tol:
            sigma2_v_previous = sigma2_v
            if self.method == "ML":  # Fisher-scoring algorithm
                deriv_sigma, info_sigma = self._derivatives_sigma(
                    area=area,
                    yhat=yhat,
                    X=X,
                    sigma2_e=sigma2_e,
                    sigma2_v=sigma2_v,
                    b_const=b_const,
                )
            elif self.method == "REML":  # Fisher-scoring algorithm
                deriv_sigma, info_sigma = self._derivatives_sigma(
                    area=area,
                    yhat=yhat,
                    X=X,
                    sigma2_e=sigma2_e,
                    sigma2_v=sigma2_v,
                    b_const=b_const,
                )
            elif self.method == "FH":  # Fisher-scoring algorithm
                deriv_sigma, info_sigma = self._derivatives_sigma(
                    area=area,
                    yhat=yhat,
                    X=X,
                    sigma2_e=sigma2_e,
                    sigma2_v=sigma2_v,
                    b_const=b_const,
                )
            else:
                AssertionError("fiting_method must be specified")

            sigma2_v += deriv_sigma / info_sigma
            sigma2_v_cov = 1 / info_sigma
            if rtol == 0:
                tolerance = abs(sigma2_v - sigma2_v_previous)
            else:  # rtol > 0:
                tolerance = abs(sigma2_v - sigma2_v_previous) / sigma2_v_previous

            convergence = tolerance <= tol
            if iterations == maxiter:
                break
            else:
                iterations += 1

        sigma2_v = float(max(sigma2_v, 0))

        return sigma2_v, sigma2_v_cov, iterations, tolerance, convergence

    def _eb_estimates(
        self,
        area: np.ndarray,
        yhat: np.ndarray,
        X: np.ndarray,
        beta: np.ndarray,
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

        v_i = sigma2_e + sigma2_v * b_const ** 2
        V = np.diag(v_i)
        V_inv = np.diag(1 / v_i)
        mu = np.matmul(X, beta)
        resid = yhat - mu
        G = np.diag(np.ones(np.size(yhat)) * sigma2_v)

        Z = np.diag(b_const)

        b = np.matmul(np.matmul(G, np.transpose(Z)), V_inv)
        d = np.transpose(X - np.matmul(np.transpose(b), X))
        x_vinv_x = np.matmul(np.matmul(np.transpose(X), V_inv), X)
        g2_term = np.linalg.inv(x_vinv_x)

        zTz = np.matmul(np.matmul(np.transpose(Z), V_inv), Z)
        zTz_diag = np.diag(zTz)
        b_term_ml1 = np.sum(zTz_diag)
        b_term_ml2 = np.sum(zTz_diag * b_const ** 2 / v_i)

        b_term_ml1 = np.linalg.inv(x_vinv_x)
        b_term_ml2_diag = b_const ** 2 / v_i ** 2
        b_term_ml2 = np.matmul(np.matmul(np.transpose(X), np.diag(b_term_ml2_diag)), X)
        b_term_ml = np.trace(np.matmul(b_term_ml1, b_term_ml2))

        b_term_fh1 = np.size(yhat) * np.sum(1 / v_i ** 2)
        b_term_fh2 = np.sum(1 / v_i) ** 2
        b_term_fh3 = np.sum(1 / v_i) ** 3
        b_term_fh = 2 * (b_term_fh1 - b_term_fh2) / b_term_fh3

        estimates = np.array(yhat) * np.nan
        g1 = np.array(yhat) * np.nan
        g2 = np.array(yhat) * np.nan
        g3 = np.array(yhat) * np.nan
        g3_star = np.array(yhat) * np.nan

        g1_partial = np.array(yhat) * np.nan

        for d in area:
            b_d = b_const[area == d]
            phi_d = sigma2_e[area == d]
            X_d = X[area == d]
            yhat_d = yhat[area == d]
            mu_d = np.matmul(X_d, beta)
            resid_d = yhat_d - mu_d
            variance_d = sigma2_v * b_d ** 2 + phi_d
            gamma_d = sigma2_v * b_d ** 2 / variance_d
            estimates[area == d] = gamma_d * yhat_d + (1 - gamma_d) * mu_d
            g1[area == d] = gamma_d * phi_d
            g2_term_d = np.matmul(np.matmul(X_d, g2_term), np.transpose(X_d))
            g2[area == d] = (1 - gamma_d) ** 2 * float(g2_term_d)
            g3[area == d] = (phi_d ** 2 * b_d ** 4 / variance_d ** 3) * sigma2_v_cov
            g3_star[area == d] = (g3[area == d] / variance_d) * resid_d ** 2
            if self.method == "ML":
                b_sigma2_v = -(1 / 2 * sigma2_v_cov) * b_term_ml
            elif self.method == "FH":
                b_sigma2_v = -(1 / 2 * sigma2_v_cov) * b_term_fh
            else:
                b_sigma2_v = 0
            g1_partial[area == d] = b_d ** 2 * (1 - gamma_d) ** 2 * b_sigma2_v

        if self.method == "REML":
            mse = g1 + g2 + 2 * g3
            mse1_area_specific = g1 + g2 + 2 * g3_star
            mse2_area_specific = g1 + g2 + g3 + g3_star
        elif self.method in ("ML", "FH"):
            mse = g1 - g1_partial + g2 + 2 * g3
            mse1_area_specific = g1 - g1_partial + g2 + 2 * g3_star
            mse2_area_specific = g1 - g1_partial + g2 + g3 + g3_star

        return (estimates, mse, mse1_area_specific, mse2_area_specific, g1, g2, g3, g3_star)

    def fit(
        self,
        area: Array,
        yhat: Array,
        X: Array,
        sigma2_e: Array,
        method: str = "REML",
        b_const: Union[np.array, float, int] = 1.0,
        sigma2_v_start: float = 0.001,
        maxiter: int = 100,
        tol: float = 1.0e-6,
        rtol: float = 0.0,
        to_dataframe: bool = False,
    ) -> None:

        if method.upper() not in ("FH", "ML", "REML"):
            raise AssertionError("method must be 'FH', 'ML, or 'REML'.")
        else:
            self.method = method.upper()

        if isinstance(b_const, (int, float)):
            b_const = np.ones(area.size) * b_const

        (sigma2_v, sigma2_v_cov, iterations, tolerance, convergence) = self._iterative_methods(
            area=area,
            yhat=yhat,
            X=X,
            sigma2_e=sigma2_e,
            b_const=b_const,
            sigma2_v_start=sigma2_v_start,
            maxiter=maxiter,
            tol=tol,
            rtol=rtol,
        )

        beta, beta_cov = self._fixed_coefs(
            area=area, yhat=yhat, X=X, sigma2_e=sigma2_e, sigma2_v=sigma2_v, b_const=b_const
        )

        self.fe_coef["beta"] = beta
        self.fe_coef["stderr"] = beta_cov
        self.fe_coef["tvalue"] = None
        self.fe_coef["pvalue"] = None

        self.re_coef["sigma2"] = sigma2_v
        self.re_coef["stderr"] = sigma2_v_cov

        self.convergence = convergence
        self.iterations = iterations
        self.tolerance = tolerance

        self.goodness["AIC"] = None
        self.goodness["BIC"] = None
        self.goodness["KIC"] = None

        if to_dataframe:
            self.estimates_df = pd.DataFrame.from_dict(self.estimates)
            self.fe_coef_df = pd.DataFrame.from_dict(self.fe_coef)
            self.re_coef_df = pd.DataFrame.from_dict(self.re_coef)
            self.method_df = pd.DataFrame.from_dict(self.method)
            self.goodness_df = pd.DataFrame.from_dict(self.goodness)

    def predict(self, link: str = None) -> Tuple[Dict[Any, float], Dict[Any, float]]:

        if self.estimates is None:
            AssertionError("Check that the model was fitted before running the predictions")

        point_est, mse, mse1, mse2, g1, g2, g3, g3_star = self._eb_estimates(
            domains=self.domains,
            fitting_method=self.fitting_method["method"],
            y=self.survey_estimates,
            X=self.fe_covariates,
            beta=self.fixed_effects["beta"],
            sampling_errors=self.sampling_errors,
            sigma2_v=self.random_effects["sigma2"],
            sigma2_v_cov=self.random_effects["stderr"],
        )

        self.point_est = dict()

