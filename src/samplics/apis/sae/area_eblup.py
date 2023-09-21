"""EBLUP Area Model

"""

import math

import numpy as np

from samplics.types import (
    Array,
    AuxVars,
    DictStrNum,
    DirectEst,
    EblupEst,
    EblupFit,
    FitMethod,
    Number,
)
from samplics.utils.formats import numpy_array


# Fitting a EBLUP model
def fit_eblup(
    y: DirectEst,
    x: AuxVars,
    method: FitMethod,
    intercept: bool = True,  # if True, it adds an intercept of 1
    err_init: float | None = None,
    b_const: Array | Number = 1.0,
    tol: float = 1e-4,
    maxiter: int = 100,
) -> EblupFit:
    # TODO: add a test to check that area is the same in y and x

    area = y.to_numpy(keep_vars="__domain").flatten()
    yhat = y.to_numpy(keep_vars="est").flatten()
    if intercept:
        X = np.insert(
            x.to_numpy(drop_vars=["__record_id", "__domain"]), 0, 1, axis=1
        )  # add the intercept
    else:
        X = x.to_numpy(drop_vars=["__record_id", "__domain"])

    error_std = y.to_numpy(keep_vars="stderr").flatten()

    if (error_std <= 0).any():
        raise ValueError(
            """Some of the standard errors are not strictly positive. All standard errors must be greater than 0."""
        )

    if isinstance(b_const, (int, float)):
        b_const = np.asarray(np.ones(area.size) * b_const)
    else:
        b_const = numpy_array(b_const)

    if err_init is None:
        err_init = np.median(error_std)

    (
        sigma2_v,
        sigma2_v_cov,
        iterations,
        tolerance,
        convergence,
    ) = _iterative_fisher_scoring(
        method=method,
        area=area,
        yhat=yhat,
        X=X,
        sigma2_e=error_std**2,
        b_const=b_const,
        sigma2_v_start=err_init**2,
        tol=tol,
        maxiter=maxiter,
    )

    beta, beta_cov = _fixed_coefficients(
        area=area,
        yhat=yhat,
        X=X,
        sigma2_e=error_std**2,
        sigma2_v=sigma2_v,
        b_const=b_const,
    )

    yhat = yhat
    error_std = error_std
    X = X
    area = area

    m = yhat.size
    p = X.shape[1]
    Z_b2_Z = np.ones(shape=(m, m))
    V = np.diag(error_std**2) + sigma2_v * Z_b2_Z

    log_llike = _log_likelihood(method=method, y=yhat, X=X, beta=beta, V=V)

    return EblupFit(
        method=method,
        err_stderr=dict(zip(area, error_std)),
        fe_est=tuple(beta),
        fe_stderr=tuple(np.diag(beta_cov) ** (1 / 2)),
        re_stderr=sigma2_v ** (1 / 2),
        re_stderr_var=sigma2_v_cov,
        log_llike=log_llike,
        convergence={
            "achieved": convergence,
            "iterations": iterations,
            "precision": tolerance,
        },
        goodness={
            "AIC": -2 * log_llike + 2 * (p + 1),
            "BIC": 2 * log_llike + math.log(m) * (p + 1),
        },
    )


def _iterative_fisher_scoring(
    method: FitMethod,
    area: np.ndarray,
    yhat: np.ndarray,
    X: np.ndarray,
    sigma2_e: np.ndarray,
    b_const: np.ndarray,
    sigma2_v_start: float,
    tol: float,
    maxiter: int,
) -> tuple[float, float, int, float, bool]:  # May not need variance
    """Fisher-scroring algorithm for estimation of variance component
    return (sigma, covariance, number_iterations, tolerance, covergence status)"""

    iterations = 0
    tolerance = tol + 1.0
    sigma2_v_previous = sigma2_v_start
    info_sigma = 0.0
    while iterations < maxiter and tolerance > tol:
        deriv_sigma, info_sigma = _partial_derivatives(
            method=method,
            area=area,
            yhat=yhat,
            X=X,
            sigma2_e=sigma2_e,
            sigma2_v=sigma2_v_previous,
            b_const=b_const,
        )
        sigma2_v = sigma2_v_previous + deriv_sigma / info_sigma
        # print(tolerance)
        # if iterations == 11:
        #     breakpoint()
        # print(iterations)
        tolerance = abs((sigma2_v - sigma2_v_previous) / sigma2_v_previous)
        iterations += 1
        sigma2_v_previous = sigma2_v

    return (
        float(max(sigma2_v, 0)),
        1 / info_sigma,
        iterations,
        tolerance,
        tolerance <= tol,
    )


def _partial_derivatives(
    method: FitMethod,
    area: np.ndarray,
    yhat: np.ndarray,
    X: np.ndarray,
    sigma2_e: np.ndarray,
    sigma2_v: Number,
    b_const: np.ndarray,
) -> tuple[Number, Number]:
    deriv_sigma = 0.0
    info_sigma = 0.0
    if method == FitMethod.ml:
        beta, beta_cov = _fixed_coefficients(
            area=area,
            yhat=yhat,
            X=X,
            sigma2_e=sigma2_e,
            sigma2_v=sigma2_v,
            b_const=b_const,
        )
        for d in area:
            b_d = b_const[area == d][0]
            phi_d = sigma2_e[area == d][0]
            X_d = X[area == d, :]
            yhat_d = yhat[area == d][0]
            mu_d = np.matmul(X_d, beta)[0]
            resid_d = yhat_d - mu_d
            sigma2_d = sigma2_v * (b_d**2) + phi_d
            term1 = b_d**2 / sigma2_d
            term2 = ((b_d**2) * (resid_d**2)) / (sigma2_d**2)
            deriv_sigma += -0.5 * (term1 - term2)
            info_sigma += 0.5 * (term1**2)
    elif method == FitMethod.reml:
        B = np.diag(b_const**2)
        # v_i = sigma2_e + sigma2_v * (b_const**2)
        # V = np.diag(v_i)
        # v_inv = np.linalg.inv(V)
        v_i = 1 / (sigma2_e + sigma2_v * (b_const**2))
        v_inv = np.diag(v_i)
        x_vinv_x = np.transpose(X) @ v_inv @ X
        x_xvinvx_x = X @ np.linalg.inv(x_vinv_x) @ np.transpose(X)
        P = v_inv - v_inv @ x_xvinvx_x @ v_inv
        Py = P @ yhat
        PB = P @ B
        # P_B_P = P_B @ P
        term1 = -0.5 * np.trace(PB)
        term2 = 0.5 * (np.transpose(Py) @ B @ Py)
        deriv_sigma = term1 + term2
        info_sigma = 0.5 * np.trace(PB @ PB)
    elif method == FitMethod.fh:  # Fay-Herriot approximation
        beta, beta_cov = _fixed_coefficients(
            area=area,
            yhat=yhat,
            X=X,
            sigma2_e=sigma2_e,
            sigma2_v=sigma2_v,
            b_const=b_const,
        )
        for d in area:
            b_d = b_const[area == d]
            phi_d = sigma2_e[area == d]
            X_d = X[area == d, :]
            yhat_d = yhat[area == d]
            mu_d = np.dot(X_d, beta)
            resid_d = yhat_d - mu_d
            sigma2_d = sigma2_v * (b_d**2) + phi_d
            deriv_sigma += float((resid_d**2) / sigma2_d)
            info_sigma += -float(((b_d**2) * (resid_d**2)) / (sigma2_d**2))
        m = yhat.size
        p = X.shape[1]
        deriv_sigma = m - p - deriv_sigma

    return deriv_sigma, info_sigma


def _fixed_coefficients(
    area: np.ndarray,
    yhat: np.ndarray,
    X: np.ndarray,
    sigma2_e: np.ndarray,
    sigma2_v: float,
    b_const: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # V = np.diag(sigma2_v * (b_const**2) + sigma2_e)
    # V_inv = np.linalg.inv(V)
    V_inv = np.diag(1 / (sigma2_v * (b_const**2) + sigma2_e))
    x_v_X_inv = np.linalg.inv(np.matmul(np.matmul(np.transpose(X), V_inv), X))
    x_v_x_inv_x = np.matmul(np.matmul(x_v_X_inv, np.transpose(X)), V_inv)
    beta_hat = np.matmul(x_v_x_inv_x, yhat)
    beta_cov = np.matmul(np.matmul(np.transpose(X), V_inv), X)

    return beta_hat.ravel(), np.linalg.inv(beta_cov)


def _log_likelihood(
    method: FitMethod, y: np.ndarray, X: np.ndarray, beta: np.ndarray, V: np.ndarray
) -> Number:
    m = y.size
    const = m * np.log(2 * np.pi)
    ll_term1 = np.log(np.linalg.det(V))
    V_inv = np.linalg.inv(V)
    resid_term = y - np.dot(X, beta)
    if method in (FitMethod.ml, FitMethod.fh):  # What is likelihood for FH
        resid_var = np.dot(np.transpose(resid_term), V_inv)
        ll_term2 = np.dot(resid_var, resid_term)
        loglike = -0.5 * (const + ll_term1 + ll_term2)
    elif method == FitMethod.reml:
        xT_vinv_x = np.dot(np.dot(np.transpose(X), V_inv), X)
        ll_term2 = np.log(np.linalg.det(xT_vinv_x))
        ll_term3 = np.dot(np.dot(y, V_inv), resid_term)
        loglike = -0.5 * (const + ll_term1 + ll_term2 + ll_term3)
    else:
        raise AssertionError("A fitting method must be specified.")

    return float(loglike)


def predict_eblup(
    fit_eblup: EblupFit,
    y: DirectEst,
    x: AuxVars,
    b_const: DictStrNum | Number = 1.0,
) -> EblupEst:
    area = numpy_array(x.area)
    X = np.insert(x.to_numpy(), 0, 1, axis=1)

    if isinstance(b_const, (int, float)):
        b_const = dict(zip(area, np.ones(area.size) * b_const))

    sigme2_e = {}
    for d in fit_eblup.e_stderr:
        sigme2_e[d] = fit_eblup.e_stderr[d] ** 2

    breakpoint()
    est, mse, mse1, mse2, g1, g2, g3, g3_star = _eb_estimates(
        method=fit_eblup,
        yhat=y.est,
        X=X,
        area=np.array(area),
        beta=np.array(fit_eblup.fe_est),
        sigma2_e=sigme2_e,
        sigma2_v=fit_eblup.re_stderr**2,
        sigma2_v_cov=fit_eblup.re_stderr_var,
        b_const=b_const,
    )

    # self.est = dict(zip(area, point_est))
    # self.mse = dict(zip(area, mse))

    return EblupEst(area=area, est=dict(zip(area, est)), mse=dict(zip(area, mse)))


def _eb_estimates(
    method: FitMethod,
    yhat: dict,
    X: np.ndarray,
    beta: np.ndarray,
    area: np.ndarray,
    sigma2_e: dict,
    sigma2_v: Number,
    sigma2_v_cov: Number,
    b_const: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    m = len(yhat)
    v_i = sigma2_e + sigma2_v * (b_const**2)
    V_inv = np.diag(1 / v_i)
    G = np.diag(np.ones(m) * sigma2_v)

    Z = np.diag(b_const)

    b = np.matmul(np.matmul(G, np.transpose(Z)), V_inv)
    d = np.transpose(X - np.matmul(np.transpose(b), X))
    x_vinv_x = np.matmul(np.matmul(np.transpose(X), V_inv), X)

    g2_term = np.linalg.inv(x_vinv_x)

    b_term_ml1 = np.linalg.inv(x_vinv_x)
    b_term_ml2_diag = (b_const**2) / (v_i**2)
    b_term_ml2 = np.matmul(np.matmul(np.transpose(X), np.diag(b_term_ml2_diag)), X)
    b_term_ml = float(np.trace(np.matmul(b_term_ml1, b_term_ml2)))

    estimates = {}  # np.zeros(m) * np.nan
    g1 = {}  # np.zeros(m) * np.nan
    g2 = {}  # np.zeros(m) * np.nan
    g3 = {}  # np.zeros(m) * np.nan
    g3_star = {}  # np.zeros(m) * np.nan

    g1_partial = {}  # np.zeros(m) * np.nan

    sum_inv_vi2 = np.sum(1 / (v_i**2))
    b_sigma2_v = 0.0
    if method == FitMethod.reml:
        g3_scale = 2.0 / sum_inv_vi2
    elif method == FitMethod.ml:
        b_sigma2_v = -(1.0 / 2.0 * sigma2_v_cov) * b_term_ml
        g3_scale = 2.0 / sum_inv_vi2
    elif method == FitMethod.fh:
        sum_vi = np.sum((1 / v_i))
        b_sigma2_v = 2.0 * (m * sum_inv_vi2 - sum_vi**2) / (sum_vi**3)
        g3_scale = 2.0 * m / sum_vi**2
    else:
        g3_scale = 0.0

    breakpoint()

    for d in area:
        b_d = b_const[d]
        phi_d = sigma2_e[d]
        X_d = X[area == d]
        yhat_d = yhat[d]
        mu_d = np.matmul(X_d, beta)
        resid_d = yhat_d - mu_d
        variance_d = sigma2_v * (b_d**2) + phi_d
        gamma_d = sigma2_v * (b_d**2) / variance_d
        estimates[area == d] = gamma_d * yhat_d + (1 - gamma_d) * mu_d
        g1[area == d] = gamma_d * phi_d
        g2_term_d = np.matmul(np.matmul(X_d, g2_term), np.transpose(X_d))
        g2[area == d] = ((1 - gamma_d) ** 2) * float(g2_term_d)
        g3[area == d] = ((1 - gamma_d) ** 2) * g3_scale / variance_d
        g3_star[area == d] = (g3[area == d] / variance_d) * (resid_d**2)
        g1_partial[area == d] = (b_d**2) * ((1 - gamma_d) ** 2) * b_sigma2_v

    mse = 0
    mse1_area_specific = 0
    mse2_area_specific = 0
    if method == FitMethod.reml:
        mse = g1 + g2 + 2 * g3
        mse1_area_specific = g1 + g2 + 2 * g3_star
        mse2_area_specific = g1 + g2 + g3 + g3_star
    elif method in (FitMethod.fh, FitMethod.ml):
        mse = g1 - g1_partial + g2 + 2 * g3
        mse1_area_specific = g1 - g1_partial + g2 + 2 * g3_star
        mse2_area_specific = g1 - g1_partial + g2 + g3 + g3_star

    return (
        np.asarray(estimates),
        np.asarray(mse),
        np.asarray(mse1_area_specific),
        np.asarray(mse2_area_specific),
        np.asarray(g1),
        np.asarray(g2),
        np.asarray(g3),
        np.asarray(g3_star),
    )


#######################


# import math

# from typing import Any

# import numpy as np
# import pandas as pd

# from samplics.utils.formats import dict_to_dataframe, numpy_array
# from samplics.utils.types import Array, DictStrNum, Number


# class EblupAreaModel:
#     def __init__(self, method: str = "REML") -> None:
#         if method.upper() not in ("FH", "ML", "REML"):
#             raise AssertionError("Parameter method must be 'FH', 'ML, or 'REML'.")
#         else:
#             self.method = method.upper()

#         # Sample data
#         self.yhat: np.ndarray
#         self.error_std: np.ndarray
#         self.X: np.ndarray
#         self.area: np.ndarray

#         # Fitting stats
#         self.fitted: bool = False
#         self.fixed_effects: np.ndarray
#         self.fe_std: np.ndarray
#         self.re_std: Number
#         self.convergence: dict[str, Union[float, int, bool]] = {}
#         self.goodness: dict[str, Number] = {}  # loglikehood, deviance, AIC, BIC

#         # Predict(ino/ed) data
#         self.est: DictStrNum
#         self.area_mse: DictStrNum
#         self.area_mse_as1: DictStrNum
#         self.area_mse_as2: DictStrNum
#         self.area_mse_terms: dict[str, DictStrNum]

#     def __str__(self) -> str:
#         estimation = pd.DataFrame()
#         estimation["area"] = self.area
#         estimation["estimate"] = self.est
#         estimation["mse"] = self.est

#         fit = pd.DataFrame()
#         fit["beta_coef"] = self.fixed_effects
#         fit["beta_stderr"] = np.diag(self.fe_std)

#         return f"""\n\nFH Area Model - Best predictor,\n\nConvergence status: {self.convergence['achieved']}\nNumber of iterations: {self.convergence['iterations']}\nPrecision: {self.convergence['precision']}\n\nGoodness of fit: {self.goodness}\n\nEstimation:\n{estimation}\n\nFixed effect:\n{fit}\n\nRandom effect variance:\n{self.re_std**2}\n\n"""

#     def __repr__(self) -> str:
#         return self.__str__()


#     def to_dataframe(self, col_names: Optional(list) = None) -> pd.DataFrame:
#         """Returns a pandas dataframe from dictionaries with same keys and one value per key.

#         Args:
#             col_names (list, optional): list of string to be used for the dataframe columns names.
#                 Defaults to ["_parameter", "_area", "_estimate", "_mse"].

#         Returns:
#             [type]: a pandas dataframe
#         """

#         est_df = dict_to_dataframe(col_names, self.est, self.area_mse)
#         est_df.iloc[:, 0] = "mean"

#         return est_df
