from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import math

from scipy import linalg
from scipy.stats import norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


def area_stats(
    y: np.ndarray,
    X: np.ndarray,
    area: np.ndarray,
    error_std: float,
    re_std: float,
    a_factor: Dict[Any, float],
    samp_weight: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    areas = np.unique(area)
    y_mean = np.zeros(areas.size)
    X_mean = np.zeros((areas.size, X.shape[1]))
    gamma = np.zeros(areas.size)
    samp_size = np.zeros(areas.size)
    for k, d in enumerate(areas):
        sample_d = area == d
        a_factor_d = a_factor[d]
        if samp_weight is None:
            weight_d = np.ones(np.sum(sample_d))
            delta_d = 1 / np.sum(a_factor_d)
        else:
            weight_d = samp_weight[sample_d]
            delta_d = np.sum((weight_d / np.sum(weight_d)) ** 2)
        aw_factor_d = weight_d * a_factor_d
        yw_d = y[sample_d] * a_factor_d
        y_mean[k] = np.sum(yw_d) / np.sum(aw_factor_d)
        Xw_d = X[sample_d, :] * aw_factor_d[:, None]
        X_mean[k, :] = np.sum(Xw_d, axis=0) / np.sum(aw_factor_d)
        gamma[k] = (re_std ** 2) / (re_std ** 2 + (error_std ** 2) * delta_d)
        samp_size[k] = np.sum(sample_d)

    return y_mean, X_mean, gamma, samp_size.astype(int)


def fixed_coefficients(
    area: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    sigma2e: float,
    sigma2u: float,
    scale: np.ndarray,
) -> np.ndarray:  # Tuple[np.ndarray, np.ndarray]:

    """[summary]    Arguments:

        area {np.ndarray} -- [description]
        y {np.ndarray} -- [description]
        X {np.ndarray} -- [description]
        sigma2e {np.ndarray} -- [description]
        sigma2u {float} -- [description]
        scale {np.ndarray} -- [description]
    Returns:
        Tuple[np.ndarray, np.ndarray] -- [description]
    """
    V_inv = inverse_covariance(area, sigma2e, sigma2u, scale)
    x_v_X_inv = np.linalg.inv(np.matmul(np.matmul(np.transpose(X), V_inv), X))
    x_v_x_inv_x = np.matmul(np.matmul(x_v_X_inv, np.transpose(X)), V_inv)
    beta_hat = np.matmul(x_v_x_inv_x, y)

    # beta_hat_cov = np.matmul(np.matmul(np.transpose(X), V_inv), X)
    return beta_hat.ravel()  # , np.linalg.inv(beta_hat_cov)


def covariance(area: np.ndarray, sigma2e: float, sigma2u: float, scale: np.ndarray,) -> np.ndarray:

    n = area.shape[0]
    areas, areas_size = np.unique(area, return_counts=True)
    V = np.zeros((n, n))
    for i, d in enumerate(areas):
        nd, scale_d = areas_size[i], scale[area == d]
        Rd = sigma2e * np.diag(scale_d)
        Zd = np.ones((nd, nd))
        start = 0 if i == 0 else sum(areas_size[:i])
        end = n if i == areas.size - 1 else sum(areas_size[: (i + 1)])
        V[start:end, start:end] = Rd + sigma2u * Zd

    return V


def inverse_covariance(
    area: np.ndarray, sigma2e: float, sigma2u: float, scale: np.ndarray,
) -> np.ndarray:

    n = area.shape[0]
    areas, areas_size = np.unique(area, return_counts=True)
    V_inv = np.zeros((n, n))
    for i, d in enumerate(areas):
        nd, scale_d = areas_size[i], scale[area == d]
        a_d = 1 / (scale_d ** 2)
        sum_scale_d = np.sum(a_d)
        Zd = np.ones((nd, nd))
        start = 0 if i == 0 else sum(areas_size[:i])
        end = n if i == areas.size - 1 else sum(areas_size[: (i + 1)])
        gamma_d = sigma2u / (sigma2u + sigma2e / sum_scale_d)
        V_inv[start:end, start:end] = (1 / sigma2e) * (
            np.diag(a_d)
            - (gamma_d / sum_scale_d) * np.matmul(a_d[:, None], np.transpose(a_d[:, None]))
        )

    return V_inv


def log_det_covariance(
    area: np.ndarray, sigma2e: float, sigma2u: float, scale: np.ndarray
) -> float:

    det = 0
    for d in np.unique(area):
        nd = np.sum(area == d)
        det += np.sum(np.log(scale)) + nd * np.log(sigma2e) + np.log(1 + nd * sigma2u / sigma2e)

    return det


def log_likelihood(
    method: str,
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    inv_covariance: np.ndarray,
    log_det_covariance: float,
) -> float:

    n = y.size
    const = n * np.log(2 * np.pi)
    ll_term1 = log_det_covariance  # np.log(np.linalg.det(covariance))
    resid_term = y - np.dot(X, beta)
    if method == "ML":
        resid_var = np.dot(np.transpose(resid_term), inv_covariance)
        ll_term2 = np.dot(resid_var, resid_term)
        loglike = -0.5 * (const + ll_term1 + ll_term2)
    elif method == "REML":
        xT_vinv_x = np.dot(np.dot(np.transpose(X), inv_covariance), X)
        ll_term2 = np.log(np.linalg.det(xT_vinv_x))
        ll_term3 = np.dot(np.dot(y, inv_covariance), resid_term)
        loglike = -0.5 * (const + ll_term1 + ll_term2 + ll_term3)
    else:
        raise AssertionError("A fitting method must be specified.")

    return float(loglike)


def partial_derivatives(
    method: str,
    area: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    # beta: np.ndarray,
    sigma2e: float,
    sigma2u: float,
    scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    derivatives = np.zeros(2)
    info_matrix = np.zeros((2, 2))

    if method == "ML":
        beta = fixed_coefficients(area, y, X, sigma2e, sigma2u, scale)
        for d in np.unique(area):
            aread = area == d
            area_d = area[aread]
            y_d = y[aread]
            X_d = X[aread]
            scale_d = scale[aread]
            nd = np.sum(aread)
            V_inv = inverse_covariance(area_d, sigma2e, sigma2u, scale_d)
            V_e = np.diag((scale_d ** 2))
            V_u = np.ones((nd, nd))
            V_inv_e = -np.matmul(V_inv @ V_e, V_inv)
            V_inv_u = -np.matmul(V_inv @ V_u, V_inv)
            error_term = y_d - np.matmul(X_d, beta)

            derivatives[0] = (
                derivatives[0]
                - 0.5 * np.trace(V_inv @ V_e)
                - 0.5 * error_term @ V_inv_e @ error_term
            )

            derivatives[1] = (
                derivatives[1]
                - 0.5 * np.trace(V_inv @ V_u)
                - 0.5 * error_term @ V_inv_u @ error_term
            )

            info_matrix[0, 0] = info_matrix[0, 0] + 0.5 * np.trace(
                np.matmul(V_inv, V_e) @ np.matmul(V_inv, V_e)
            )
            info_matrix[1, 1] = info_matrix[1, 1] + 0.5 * np.trace(
                np.matmul(V_inv, V_u) @ np.matmul(V_inv, V_u)
            )
            info_matrix[0, 1] = info_matrix[0, 1] + 0.5 * np.trace(
                np.matmul(V_inv, V_e) @ np.matmul(V_inv, V_u)
            )
            info_matrix[1, 0] = info_matrix[0, 1]

    elif method == "REML":
        n = y.shape[0]
        V_inv = inverse_covariance(area, sigma2e, sigma2u, scale)
        V_e = np.diag((scale ** 2))
        V_u = np.ones((n, n))
        x_vinv_x = np.transpose(X) @ V_inv @ X
        x_xvinvx_x = X @ np.linalg.inv(x_vinv_x) @ np.transpose(X)
        P = V_inv - V_inv @ x_xvinvx_x @ V_inv

        derivatives[0] = -0.5 * np.trace(P @ V_e) + 0.5 * y @ P @ V_e @ P @ y
        derivatives[1] = -0.5 * np.trace(P @ V_u) + 0.5 * y @ P @ V_u @ P @ y

        info_matrix[0, 0] = 0.5 * np.trace(np.matmul(P, V_e) @ np.matmul(P, V_e))
        info_matrix[1, 1] = 0.5 * np.trace(np.matmul(P, V_u) @ np.matmul(P, V_u))
        info_matrix[0, 1] = 0.5 * np.trace(np.matmul(P, V_e) @ np.matmul(P, V_u))
        info_matrix[1, 0] = info_matrix[0, 1]

    return derivatives, info_matrix


def iterative_fisher_scoring(
    method: str,
    area: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    sigma2e: float,
    sigma2u: float,
    scale: np.ndarray,
    abstol: float,
    reltol: float,
    maxiter: int,
) -> Tuple[float, float, int, float, bool]:  # May not need variance
    """ Fisher-scroring algorithm for estimation of variance component"""

    iterations = 0
    tolerance, tol = 1, 0.9
    sigma2 = np.asarray([0, 0])
    sigma2_previous = np.asarray([0, 0])
    while tolerance > tol:
        derivatives, info_matrix = partial_derivatives(
            method, area=area, y=y, X=X, sigma2e=sigma2e, sigma2u=sigma2u, scale=scale,
        )

        # print(np.matmul(np.linalg.inv(info_matrix), derivatives))
        sigma2 = sigma2 + derivatives @ np.linalg.inv(info_matrix)
        sigma2e, sigma2u = sigma2[0], sigma2[1]
        print(tolerance)
        tolerance = min(abs(sigma2 - sigma2_previous))
        tol = max(abstol, reltol * min(abs(sigma2)))
        convergence = tolerance <= tol
        sigma2_previous = sigma2

        if iterations == maxiter:
            break
        else:
            iterations += 1

    return sigma2, np.linalg.inv(info_matrix), iterations, tolerance, convergence
