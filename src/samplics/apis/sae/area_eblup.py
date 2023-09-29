"""EBLUP Area Model

"""

import math

import numpy as np
import polars as pl

from samplics.types import (
    AuxVars,
    DictStrNum,
    DirectEst,
    EblupEst,
    EblupFit,
    FitMethod,
    GlmmFitStats,
    Mse,
    Number,
)


# Fitting a EBLUP model
def fit_eblup(
    y: DirectEst,
    x: AuxVars,
    method: FitMethod,
    intercept: bool = True,  # if True, it adds an intercept of 1
    err_init: float | None = None,
    # b_const: Array | Number = 1.0, #TODO: move it to the y (add it in DepVars, etc.)
    tol: float = 1e-4,
    maxiter: int = 100,
) -> EblupFit:
    # TODO: add a test to check that area is the same in y and x

    if err_init is None:
        err_init = np.median(y.to_numpy(keep_vars="stderr").flatten())

    (
        sig2_v,
        sig2_v_cov,
        iterations,
        tolerance,
        convergence,
    ) = _iterative_fisher_scoring(
        method=method,
        # area=area,
        y=y,
        x=x,
        sig_e=y.stderr,
        # b_const=b_const,
        intercept=intercept,
        sig_v_start=err_init,
        tol=tol,
        maxiter=maxiter,
    )

    beta, beta_cov = _fixed_coefficients(
        # area=area,
        y=y,
        x=x,
        sig2_e=y.stderr,
        sig2_v=sig2_v,
        intercept=intercept
        # b_const=b_const,
    )

    # yhat = yhat
    # error_std = error_std
    # X = X
    # area = area

    # m = yhat.size
    m, p = x.to_numpy(drop_vars=["__drop_id", "__domain"]).shape
    # R = np.diag(np.ones(m)) * (error_std**2)
    # Z = np.diag(np.ones(m))
    # G = np.diag(np.ones(m)) * sigma2_v
    # V = R + Z * G * np.transpose(Z)

    # breakpoint()

    log_llike = _log_likelihood(
        method=method,
        y=y,
        x=x,
        beta=beta,
        sig2_e=y.stderr,
        sig2_v=sig2_v,
        intercept=intercept,
    )

    return EblupFit(
        method=method,
        err_stderr=y.stderr,
        fe_est=beta.tolist(),
        fe_stderr=(np.diag(beta_cov) ** (1 / 2)).tolist(),
        re_stderr=sig2_v ** (1 / 2),
        re_stderr_var=sig2_v_cov,
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
    # area: np.ndarray,
    y: DirectEst,
    x: AuxVars,
    sig_e: dict,
    # b_const: np.ndarray,
    intercept: bool,
    sig_v_start: float,
    tol: float,
    maxiter: int,
) -> tuple[float, float, int, float, bool]:  # May not need variance
    """Fisher-scroring algorithm for estimation of variance component
    return (sigma, covariance, number_iterations, tolerance, covergence status)"""

    iterations = 0
    tolerance = tol + 1.0
    sig2_v_prev = sig_v_start**2
    sig2_e = {d: sig_e[d] ** 2 for d in sig_e}

    info_sigma = 0.0
    beta, beta_cov = None, None
    while iterations < maxiter and tolerance > tol:
        if method == FitMethod.ml or method == FitMethod.fh:
            beta, beta_cov = _fixed_coefficients(
                y=y,
                x=x,
                sig2_e=sig2_e,
                sig2_v=sig2_v_prev,
                intercept=intercept
                # b_const=b_const,
            )

        deriv_sigma, info_sigma = _partial_derivatives(
            method=method,
            # area=area,
            y=y,
            x=x,
            sig2_e=sig2_e,
            sig2_v=sig2_v_prev,
            # b_const=b_const,
            intercept=intercept,
            beta=beta,
        )
        sig2_v = sig2_v_prev + deriv_sigma / info_sigma
        # print(tolerance)
        # if iterations == 11:
        #     breakpoint()
        # print(iterations)
        tolerance = abs((sig2_v - sig2_v_prev) / sig2_v_prev)
        iterations += 1
        sig2_v_prev = sig2_v

    return (
        float(max(sig2_v, 0)),
        1 / info_sigma,
        iterations,
        tolerance,
        tolerance <= tol,
    )


def _partial_derivatives(
    method: FitMethod,
    # area: np.ndarray,
    y: DirectEst,
    x: AuxVars,
    sig2_e: dict,
    sig2_v: Number,
    # b_const: np.ndarray,
    intercept: bool,
    beta: np.ndarray,
) -> tuple[Number, Number]:
    x = x.to_polars().drop(["__record_id"])

    if intercept:
        x0 = pl.Series("__intercept", np.ones(x.shape[0]))
        x = x.insert_at_idx(0, x0)

    m, p = x.shape
    p = p - 1  # x has an extra column

    deriv_sig = 0.0
    info_sig = 0.0

    if method == FitMethod.ml:
        for d in y.domains:
            b_d = 1  # b_const[d][0]
            x_d = x.filter(pl.col("__domain") == d).drop("__domain").to_numpy()
            y_d = y.est[d]
            resid_d = y_d - (x_d @ beta)[0]
            sig2_d = sig2_v * (b_d**2) + sig2_e[d]
            term1 = b_d**2 / sig2_d
            info_sig += term1**2
            term2 = ((b_d**2) * (resid_d**2)) / (sig2_d**2)
            deriv_sig += term1 - term2
        deriv_sig = -0.5 * deriv_sig
        info_sig = 0.5 * info_sig
    elif method == FitMethod.fh:  # Fay-Herriot approximation
        for d in y.domains:
            b_d = 1  # b_const[d][0]
            x_d = x.filter(pl.col("__domain") == d).drop("__domain").to_numpy()
            y_d = y.est[d]
            resid_d = y_d - x_d @ beta
            sig2_d = sig2_v * (b_d**2) + sig2_e[d]
            deriv_sig += float((resid_d**2) / sig2_d)
            info_sig += -float(((b_d**2) * (resid_d**2)) / (sig2_d**2))
        deriv_sig = m - p - deriv_sig
    elif method == FitMethod.reml:
        b_const = 1
        B = np.diag(np.ones(x.shape[0]))
        x_mat = x.drop("__domain").to_numpy()
        sig2_e_vec = np.array(list(sig2_e.values()))
        v_inv = np.diag(1 / (sig2_e_vec + sig2_v * (b_const**2)))
        x_vinv_x = np.transpose(x_mat) @ v_inv @ x_mat
        x_xvinvx_x = x_mat @ np.linalg.inv(x_vinv_x) @ np.transpose(x_mat)
        P = v_inv - v_inv @ x_xvinvx_x @ v_inv
        Py = P @ y.to_numpy(keep_vars="est").flatten()
        PB = P @ B
        term1 = -0.5 * np.trace(PB)
        term2 = 0.5 * (np.transpose(Py) @ B @ Py)
        deriv_sig = term1 + term2
        info_sig = 0.5 * np.trace(PB @ PB)
    else:
        raise ValueError("Fitting method not available")

    return deriv_sig, info_sig


def _fixed_coefficients(
    y: dict,
    x: dict,
    sig2_e: dict,
    sig2_v: float,
    intercept: bool,
    # b_const: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_vec = y.to_numpy(keep_vars="est").flatten()
    if intercept:
        x_mat = np.insert(
            x.to_numpy(drop_vars=["__record_id", "__domain"]), 0, 1, axis=1
        )  # add the intercept
    else:
        x_mat = x.to_numpy(drop_vars=["__record_id", "__domain"])

    b_const = 1
    sig2_e_vec = np.array(list(sig2_e.values()))
    v_inv = np.diag(1 / (sig2_v * (b_const**2) + sig2_e_vec))
    x_v_X_inv = np.linalg.pinv(np.transpose(x_mat) @ v_inv @ x_mat)
    x_v_x_inv_x = x_v_X_inv @ (np.transpose(x_mat) @ v_inv)
    beta_hat = x_v_x_inv_x @ y_vec
    beta_cov = np.transpose(x_mat) @ v_inv @ x_mat

    return beta_hat.ravel(), np.linalg.inv(beta_cov)


def _log_likelihood_fh(
    method: FitMethod,
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    sig2_e: dict,
    sig2_v: float,
    intercept: bool,
) -> Number:
    pass


def _log_likelihood_ml(
    method: FitMethod,
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    sig2_e: dict,
    sig2_v: float,
    intercept: bool,
) -> Number:
    m = len(y.est.keys())
    const = m * np.log(2 * np.pi)
    ll_term1 = np.log(np.array(list(sig2_e.values())) + sig2_v).sum()
    ll_term2 = 0

    x = x.to_polars().drop(["__record_id"])
    breakpoint()
    if intercept:
        x0 = pl.Series("__intercept", np.ones(m))
        x = x.insert_at_idx(0, x0)

    for d in y.domains:
        x_d = x.filter(pl.col("__domain") == d).drop("__domain").to_numpy()
        ll_term2 += ((y.est[d] - x_d @ beta)[0] ** 2) / (sig2_e[d] + sig2_v)

    return -0.5 * (const + ll_term1 + ll_term2)


def _log_likelihood_reml(
    method: FitMethod,
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    sig2_e: dict,
    sig2_v: float,
    intercept: bool,
) -> Number:
    pass


def _log_likelihood(
    method: FitMethod,
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    sig2_e: dict,
    sig2_v: float,
    intercept: bool,
) -> Number:
    # ll_term1 = np.log(np.array(list(sig2_e.values())) + sig2_v).sum()
    ll_term2 = 0

    x = x.to_polars().drop(["__record_id", "__domain"])

    if intercept:
        x0 = pl.Series("__intercept", np.ones(x.shape[0]))
        x = x.insert_at_idx(0, x0)

    m, p = x.shape

    b_const = 1
    sig2_e_vec = np.array(list(sig2_e.values()))
    v = np.diag(sig2_v * (b_const**2) + sig2_e_vec)
    v_inv = np.diag(1 / (sig2_v * (b_const**2) + sig2_e_vec))
    x_mat = x.to_numpy()
    y_vec = y.to_numpy(keep_vars="est").flatten()
    res = y_vec - x_mat @ beta
    ll_term1 = np.log(np.linalg.det(v))
    ll_term2 = np.transpose(res) @ v_inv @ res
    if method in (FitMethod.ml, FitMethod.fh):  # What is likelihood for FH
        const = m * np.log(2 * np.pi)
        loglike = -0.5 * (const + ll_term1 + ll_term2)
    elif method == FitMethod.reml:
        const = (m - p) * np.log(2 * np.pi)
        ll_term3 = np.log(np.linalg.det(np.transpose(x_mat) @ v_inv @ x_mat))
        loglike = -0.5 * (const + ll_term1 + ll_term2 + ll_term3)
    else:
        raise AssertionError("A fitting method must be specified.")

    return loglike


def predict_eblup(
    x: AuxVars,
    fit_eblup: GlmmFitStats,
    y: DirectEst,
    mse: Mse | list[Mse] | None = None,
    intercept: bool = True,  # if True, it adds an intercept of 1
    b_const: DictStrNum | Number = 1.0,
) -> EblupEst:
    area = y.to_numpy(keep_vars="__domain").flatten()

    if isinstance(b_const, (int, float)):
        b_const = dict(zip(area, np.ones(area.size) * b_const))

    sigme2_e = {}
    for d in fit_eblup.err_stderr:
        sigme2_e[d] = fit_eblup.err_stderr[d] ** 2

    est, mse, mse1, mse2, g1, g2, g3, g3_star = _eblup_estimates(
        method=fit_eblup.method,
        yhat=y.est,
        auxvars=x,
        area=area,
        beta=np.array(fit_eblup.fe_est),
        sigma2_e=sigme2_e,
        sigma2_v=fit_eblup.re_stderr**2,
        sigma2_v_cov=fit_eblup.re_stderr_var,
        intercept=intercept,
        b_const=b_const,
    )

    return EblupEst(
        pred=est, fit_stats=fit_eblup, domain=None, mse=mse, mse_boot=None, mse_jkn=None
    )


def _eblup_estimates(
    method: FitMethod,
    yhat: dict,
    auxvars: np.ndarray,
    beta: np.ndarray,
    area: np.ndarray,
    sigma2_e: dict,
    sigma2_v: Number,
    sigma2_v_cov: Number,
    intercept: bool,
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
    b_const_vec = np.array(list(b_const.values()))
    v_i = np.array(list(sigma2_e.values())) + sigma2_v * (b_const_vec**2)
    v_inv = np.diag(1 / v_i)
    G = np.diag(np.ones(m) * sigma2_v)
    Z = np.diag(b_const_vec)
    b = (G @ np.transpose(Z)) @ v_inv

    if intercept:
        x = np.insert(
            auxvars.to_numpy(drop_vars=["__record_id", "__domain"]), 0, 1, axis=1
        )  # add the intercept
    else:
        x = auxvars.to_numpy(drop_vars=["__record_id", "__domain"])

    d = np.transpose(x - np.transpose(b) @ x)
    x_vinv_x = np.transpose(x) @ v_inv @ x

    g2_term = np.linalg.pinv(x_vinv_x)

    b_term_ml1 = g2_term  # np.linalg.inv(x_vinv_x)
    b_term_ml2_diag = (b_const_vec**2) / (v_i**2)
    b_term_ml2 = np.matmul(np.matmul(np.transpose(x), np.diag(b_term_ml2_diag)), x)
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

    mse = {}
    mse1_area_specific = {}
    mse2_area_specific = {}

    for d in area:
        b_d = b_const[d]
        phi_d = sigma2_e[d]
        del auxvars.auxdata[d]["__record_id"]

        if intercept:
            X_d = np.insert(pl.from_dict(auxvars.auxdata[d]).to_numpy(), 0, 1, axis=1)
        else:
            X_d = pl.from_dict(auxvars.auxdata[d]).to_numpy()

        yhat_d = yhat[d]
        mu_d = X_d @ beta
        resid_d = yhat_d - mu_d
        variance_d = sigma2_v * (b_d**2) + phi_d
        gamma_d = sigma2_v * (b_d**2) / variance_d
        estimates[d] = (gamma_d * yhat_d + (1 - gamma_d) * mu_d)[0]
        g1[d] = gamma_d * phi_d
        g2_term_d = ((X_d @ g2_term) @ np.transpose(X_d))[0][0]
        g2[d] = ((1 - gamma_d) ** 2) * (g2_term_d)
        g3[d] = ((1 - gamma_d) ** 2) * g3_scale / variance_d
        g3_star[d] = ((g3[d] / variance_d) * (resid_d**2))[0]
        g1_partial[d] = (b_d**2) * ((1 - gamma_d) ** 2) * b_sigma2_v

        if method == FitMethod.reml:
            mse[d] = g1[d] + g2[d] + 2 * g3[d]
            mse1_area_specific[d] = g1[d] + g2[d] + 2 * g3_star[d]
            mse2_area_specific[d] = g1[d] + g2[d] + g3[d] + g3_star[d]
        elif method in (FitMethod.fh, FitMethod.ml):
            mse[d] = g1 - g1_partial[d] + g2[d] + 2 * g3[d]
            mse1_area_specific[d] = g1[d] - g1_partial[d] + g2[d] + 2 * g3_star[d]
            mse2_area_specific[d] = g1[d] - g1_partial[d] + g2[d] + g3[d] + g3_star[d]

    return (
        estimates,
        mse,
        mse1_area_specific,
        mse2_area_specific,
        g1,
        g2,
        g3,
        g3_star,
    )
