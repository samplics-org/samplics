from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import math

from scipy import linalg
from scipy.stats import norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


def covariance(
    area: np.ndarray, sigma2e: np.ndarray, sigma2u: float, scale: np.ndarray,
) -> np.ndarray:

    n = area.shape[0]
    areas, areas_size = np.unique(area, return_counts=True)
    V = np.zeros((n, n))
    for i, d in enumerate(areas):
        nd, scale_d = areas_size[i], scale[area == d]
        Rd = sigma2e * np.diag(scale_d)
        Zd = np.ones((nd, nd))
        start, end = i * nd, (i + 1) * nd
        V[start:end, start:end] = Rd + sigma2u * Zd

    return V


def inverse_covariance(
    area: np.ndarray, sigma2e: np.ndarray, sigma2u: float, scale: np.ndarray,
) -> np.ndarray:

    n = area.shape[0]
    areas, areas_size = np.unique(area, return_counts=True)
    V_inv = np.zeros((n, n))
    for i, d in enumerate(areas):
        nd, scale_d = areas_size[i], scale[area == d]
        a_d = 1 / (scale_d ** 2)
        sum_scale_d = np.sum(a_d)
        Zd = np.ones((nd, nd))
        start, end = i * nd, (i + 1) * nd
        gamma_d = sigma2u / (sigma2u + sigma2e / sum_scale_d)
        V_inv[start:end, start:end] = (1 / sigma2e) * (
            np.diag(a_d)
            - (gamma_d / sum_scale_d) * np.matmul(a_d[:, None], np.transpose(a_d[:, None]))
        )

    return V_inv


def fixed_coefficients(
    area: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    # sigma2e: np.ndarray,
    # sigma2u: float,
    # scale: np.ndarray,
    variance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """[summary]
    
    Arguments:
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
    V_inv = np.linalg.inv(variance)
    x_v_X_inv = np.linalg.inv(np.matmul(np.matmul(np.transpose(X), V_inv), X))
    x_v_x_inv_x = np.matmul(np.matmul(x_v_X_inv, np.transpose(X)), V_inv)
    beta_hat = np.matmul(x_v_x_inv_x, y)
    # beta_hat_cov = np.matmul(np.matmul(np.transpose(X), V_inv), X)

    return beta_hat.ravel()  # , np.linalg.inv(beta_hat_cov)


def log_likelihood(
    method, y: np.ndarray, X: np.ndarray, beta: np.ndarray, covariance: np.ndarray
) -> float:

    n = y.size
    const = n * np.log(2 * np.pi)
    ll_term1 = np.log(np.linalg.det(covariance))
    V_inv = np.linalg.inv(covariance)
    resid_term = y - np.dot(X, beta)
    if method == "ML":
        resid_var = np.dot(np.transpose(resid_term), V_inv)
        ll_term2 = np.dot(resid_var, resid_term)
        loglike = -0.5 * (const + ll_term1 + ll_term2)
    elif method == "REML":
        xT_vinv_x = np.dot(np.dot(np.transpose(X), V_inv), X)
        ll_term2 = np.log(np.linalg.det(xT_vinv_x))
        ll_term3 = np.dot(np.dot(y, V_inv), resid_term)
        loglike = -0.5 * (const + ll_term1 + ll_term2 + ll_term3)
    else:
        raise AssertionError("A fitting method must be specified.")

    return float(loglike)


def partial_derivatives(
    method,
    area: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    sigma2e: np.ndarray,
    sigma2u: np.ndarray,
    scale: np.ndarray,
) -> Tuple[float, float]:

    n = y.shape[0]
    areas, nd = np.unique(area, return_counts=True)

    V = covariance(area, sigma2e, sigma2u, scale)

    if method == "ML":
        beta = fixed_coefficients(
            area=area, y=y, X=X, sigma2e=sigma2e, sigma2u=sigma2u, scale=scale
        )
        deriv_sigma = 0.0
        info_sigma = 0.0
        for d in area:
            b_d = scale[area == d]
            phi_d = sigma2e[area == d]
            X_d = X[area == d, :]
            y_d = y[area == d]
            mu_d = np.matmul(X_d, beta)
            resid_d = y_d - mu_d
            sigma2_d = sigma2u * (b_d ** 2) + phi_d
            term1 = float(b_d ** 2 / sigma2_d)
            term2 = float(((b_d ** 2) * (resid_d ** 2)) / (sigma2_d ** 2))
            deriv_sigma += -0.5 * (term1 - term2)
            info_sigma += 0.5 * (term1 ** 2)
    elif method == "REML":
        B = np.diag(scale ** 2)
        v_i = sigma2e + sigma2u * (scale ** 2)
        V = np.diag(v_i)
        v_inv = np.linalg.inv(V)
        x_vinv_x = np.matmul(np.matmul(np.transpose(X), v_inv), X)
        x_xvinvx_x = np.matmul(np.matmul(X, np.linalg.inv(x_vinv_x)), np.transpose(X))
        P = v_inv - np.matmul(np.matmul(v_inv, x_xvinvx_x), v_inv)
        P_B = np.matmul(P, B)
        P_B_P = np.matmul(P_B, P)
        term1 = np.trace(P_B)
        term2 = np.matmul(np.matmul(np.transpose(y), P_B_P), y)
        deriv_sigma = -0.5 * (term1 - term2)
        info_sigma = 0.5 * np.trace(np.matmul(P_B_P, B))

    return float(deriv_sigma), float(info_sigma)


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
