from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import math

from scipy import linalg
from scipy.stats import norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


def fixed_coefficients(
    area: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    # sigma2e: float,
    # sigma2u: float,
    # scale: np.ndarray,
    inv_cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

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
    # V = np.diag(sigma2u * (scale ** 2) + sigma2e)
    # V_inv = np.linalg.inv(variance)
    x_v_X_inv = np.linalg.inv(np.matmul(np.matmul(np.transpose(X), inv_cov), X))
    x_v_x_inv_x = np.matmul(np.matmul(x_v_X_inv, np.transpose(X)), inv_cov)
    beta_hat = np.matmul(x_v_x_inv_x, y)

    # beta_hat_cov = np.matmul(np.matmul(np.transpose(X), V_inv), X)    return beta_hat.ravel()  # , np.linalg.inv(beta_hat_cov)


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
    method,
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
    method,
    area: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    sigma2e: float,
    sigma2u: float,
    scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    derivatives = np.zeros(2)
    info_matrix = np.zeros((2, 2))
    for d in np.unique(area):
        aread = area == d
        area_d = area[aread]
        y_d = y[aread]
        X_d = X[aread]
        scale_d = scale[aread]
        nd = np.sum(aread)
        V_inv = inverse_covariance(area_d, sigma2e, sigma2u, scale_d)
        V_e = np.diag((1 / scale_d ** 2) * (1 / sigma2e))
        V_u = sigma2u * np.ones((nd, nd))
        V_inv_e = -np.matmul(np.matmul(V_inv, V_e), V_inv)
        V_inv_u = -np.matmul(np.matmul(V_inv, V_u), V_inv)

        info_matrix_d = np.zeros((2, 2))
        if method == "ML":
            error_term = y - np.matmul(X, beta)
            term1_e = -0.5 * np.trace(np.matmul(V_e, V_u))
            term2_e = -0.5 * np.matmul(np.matmul(np.transpose(error_term), V_inv_e), error_term)
            term1_u = -0.5 * np.trace(np.matmul(V_e, V_u))
            term2_u = -0.5 * np.matmul(np.matmul(np.transpose(error_term), V_inv_u), error_term)
            info_matrix_d[0, 0] = 0.5 * np.trace(
                np.matmul(np.matmul(V_inv, V_e), np.matmul(V_inv, V_e))
            )
            info_matrix_d[1, 1] = 0.5 * np.trace(
                np.matmul(np.matmul(V_inv, V_u), np.matmul(V_inv, V_u))
            )
            info_matrix_d[0, 1] = 0.5 * np.trace(
                np.matmul(np.matmul(V_inv, V_e), np.matmul(V_inv, V_u))
            )
            info_matrix_d[1, 0] = info_matrix_d[0, 1]

        elif method == "REML":
            x_vinv_x = np.matmul(np.matmul(np.transpose(X), V_inv), X)
            x_xvinvx_x = np.matmul(np.matmul(X, np.linalg.inv(x_vinv_x)), np.transpose(X))
            P = V_inv - np.matmul(np.matmul(V_inv, x_xvinvx_x), V_inv)
            term1_e = -0.5 * np.trace(np.matmul(P, V_e))
            P_V_P_e = np.matmul(np.matmul(P, V_e), P)
            term2_e = 0.5 * np.matmul(np.matmul(np.transpose(y), P_V_P_e), y)
            term1_u = -0.5 * np.trace(np.matmul(P, V_u))
            P_V_P_u = np.matmul(np.matmul(P, V_u), P)
            term2_u = 0.5 * np.matmul(np.matmul(np.transpose(y), P_V_P_u), y)
            info_matrix_d[0, 0] = 0.5 * np.trace(np.matmul(np.matmul(P, V_e), np.matmul(P, V_e)))
            info_matrix_d[1, 1] = 0.5 * np.trace(np.matmul(np.matmul(P, V_u), np.matmul(P, V_u)))
            info_matrix_d[0, 1] = 0.5 * np.trace(np.matmul(np.matmul(P, V_e), np.matmul(P, V_u)))
            info_matrix_d[1, 0] = info_matrix_d[0, 1]

        derivatives = derivatives + np.asarray([term1_e + term2_e, term1_u + term2_u])
        info_matrix = info_matrix + info_matrix_d

    return derivatives, info_matrix


def iterative_fisher_scoring(
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

    tolerance = abstol + 1.0
    tol = 0.9 * tolerance
    while tolerance > tol:
        sigma2_u_previous = sigma2u
        deriv_sigma, info_sigma = partial_derivatives(
            area=area, y=y, X=X, sigma2e=sigma2e, sigma2u=sigma2u, scale=scale,
        )

        sigma2u += deriv_sigma / info_sigma
        sigma2_u_cov = 1 / info_sigma

        tolerance = abs(sigma2u - sigma2_u_previous)
        tol = max(abstol, reltol * abs(sigma2u))
        convergence = tolerance <= tol

        if iterations == maxiter:
            break
        else:
            iterations += 1

    sigma2u = float(max(sigma2u, 0))

    return sigma2u, sigma2_u_cov, iterations, tolerance, convergence
