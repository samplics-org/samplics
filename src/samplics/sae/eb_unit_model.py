"""EBLUP and EB Unit Models.

This module implements the basic EBLUP and EB unit level models. It also provides the ELL model. 
The functionalities are organized in classes. Each class has three main methods: *fit()*, 
*predict()* and *bootstrap_mse()*. Linear Mixed Models (LMM) are the core underlying statistical
framework used to model the hierarchical nature of the small area estimation (SAE) techniques 
implemented in this module, see McCulloch, C.E. and Searle, S.R. (2001) [#ms2001]_ for more 
details on LMM.

The *EblupUnitModel* class implements the model developed by Battese, G.E., Harter, R.M., and 
Fuller, W.A. (1988) [#bhf1988]_. The model parameters can fitted using restricted maximum 
likelihood (REML) and maximum likelihood (ML). The normality assumption of the errors is not 
necessary to predict the point estimates but is required for the taylor MSE estimation. The 
predictions takes into account sampling rates. A bootstrap MSE estimation method is also implemted 
for this class. 

The *EbUnitModel* class implements the model developed by Molina, I. and Rao, J.N.K. (2010)
[#mr2010]_. So far, only the basic approach requiring the normal distribution of the errors is 
implemented. This approach allows estimating complex indicators such as poverty indices and 
other nonlinear paramaters. The class fits the model parameters using REML or ML. To predict the 
area level indicators estimates, a Monte Carlo (MC) approach is used. MSE estimation is achieved 
using a bootstrap procedure.  

The *EllUnitModel* class implements the model Elbers, C., Lanjouw, J.O., and Lanjouw, P. (2003) 
[#ell2003]_. This method is nonparametric at its core, hence does not require normality 
assumption nor any other parametric distribution. This implementation a semiparametric and 
nonparametric are provided. In the semiparametric, the normal distribution is used to fit the
parameters and to draw the fixed-effects. 

.. [#ms2001] McCulloch, C.E.and Searle, S.R. (2001), *Generalized, Linear, Mixed Models*, 
   New York: John Wiley & Sons, Inc.
.. [#bhf1988] Battese, G.E., Harter, R.M., and Fuller, W.A. (1988). An error-components model for 
   prediction of county crop areas using survey and satellite data, *Journal of the American 
   Statistical Association*, **83**, 28-36.
.. [#mr2010] Molina, , I. and Rao, J.N.K. (2010), Small Area Estimation of Poverty Indicators, 
   *Canadian Journal of Statistics*, **38**, 369-385.
.. [#ell2003] Elbers, C., Lanjouw, J.O., and Lanjouw, P. (2003), Micro-Level Estimation of Poverty
   and Inequality. *Econometrica*, **71**, 355-364.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import warnings
import numpy as np
import pandas as pd

import math

import statsmodels.api as sm

from scipy.stats import norm as normal

from samplics.utils import checks, formats, basic_functions
from samplics.utils.types import Array, Number, StringNumber, DictStrNum

from samplics.sae.sae_core_functions import area_stats


# from samplics.sae.core_sae_functions import fixed_coefficients, iterative_fisher_scoring


class EblupUnitModel:
    """*EblupUnitModel* implements the basic unit level model for means (a linear indicator).

    *EblupUnitModel* takes the sample data as input and fits the basic linear mixed model. 
    The user can pick between restricted maximum likelihood (REML) or maximum likelihood (ML) 
    to fit the model parameters. Also, EblupUnitModel predicts the areas means and provides 
    the point and mean squared error (MSE) estimates of the empirical Bayes linear 
    unbiased (EBLUP). User can also obtain the bootstrap mse estimates of the MSE.

    Setting attributes
        | method (str): the fitting method of the model parameters which can take the possible 
        |   values restricted maximum likelihood (REML) or maximum likelihood (ML). 
        |   If not specified, "REML" is used as default.  
    
    Sample related attributes
        | ys (array): the output sample observations. 
        | Xs (ndarray): the auxiliary information. 
        | scales (array): an array of scaling parameters for the unit levels errors.
        | afactors (array): sum of the inverse squared of scale.
        | areas (array): the full vector of small areas from the sampled observations.  
        | areas_list (array): the list of small areas from the sample data.
        | samp_size (dict): the sample size per small areas from the sample. 
        | ys_mean (array): sample area means of the output variable. 
        | Xs_mean (ndarray): sample area means of the auxiliary variables.

    Model fitting attributes
        | fitted (boolean): indicates whether the model has been fitted or not. 
        | fixed_effects (array): the estimated fixed effects of the regression model. 
        | fe_std (array): the estimated standard errors of the fixed effects. 
        | random_effects (array): the estimated area level random effects.
        | re_std (number): the estimated standard error of the random effects. 
        | error_std (number): the estimated standard error of the unit level residuals. 
        | convergence (dict): a dictionnary holding the convergence status and the number of 
        |   iterations from the model fitting algorithm. 
        | goodness (dict): a dictionary holding the log-likelihood, AIC, and BIC.
        | gamma (dict): ratio of the between-area variability (re_std**2) to the total 
        |   variability (re_std**2 + error_std**2 / a_factor). 

    Prediction related attributes
        | areap (array): the list of areas for the prediction. 
        | pop_size (dict): area level population sizes. 
        | Xbar (array): population means of the auxiliary variables. 
        | number_reps (int): number of replicates for the bootstrap MSE estimation. 
        | area_est (array): area level EBLUP estimates. 
        | area_mse (array): area level taylor estimation of the MSE. 
        | area_mse_boot (array): area level bootstrap estimation of the MSE.

    Main methods
        | fit(): fits the linear mixed model to estimate the model parameters using REMl or ML
        |   methods. 
        | predict(): predicts the area level mean estimates which includes both the point   | | 
        |   estimates and the taylor MSE estimate. 
        | bootstrap_mse(): computes the area level bootstrap MSE estimates of the mean.
    """

    def __init__(
        self, method: str = "REML",
    ):
        # Setting
        self.method: str = method.upper()
        if self.method not in ("REML", "ML"):
            raise AssertionError("Value provided for method is not valid!")

        # Sample data
        self.scales: np.ndarray = np.array([])
        self.afactors: Dict[Any, float] = {}
        self.ys: np.ndarray = np.array([])
        self.Xs: np.ndarray = np.array([])
        self.areas: np.ndarray = np.array([])
        self.areas_list: np.ndarray = np.array([])
        self.samp_size: Dict[Any, int] = {}
        self.ys_mean: np.ndarray = np.array([])
        self.Xs_mean: np.ndarray = np.array([])

        # Fitting stats
        self.fitted: bool = False
        self.fixed_effects: np.ndarray = np.array([])
        self.fe_std: np.ndarray = np.array([])
        self.random_effects: np.ndarray = np.array([])
        self.re_std: float = 0
        self.error_std: float = 0
        self.convergence: Dict[str, Union[float, int, bool]] = {}
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC
        self.gamma: Dict[Any, float] = {}

        # Predict(ion/ed) data
        self.areap_list: np.ndarray = np.array([])
        self.pop_size: Dict[Any, float] = {}
        self.Xp_mean: np.ndarray = np.array([])
        self.number_reps: int = 0
        self.area_est: Dict[Any, float] = {}
        self.area_mse: Dict[Any, float] = {}
        self.area_mse_boot: Dict[Any, float] = {}

    def _beta(
        self, y: np.ndarray, X: np.ndarray, area: np.ndarray, weight: np.ndarray,
    ) -> np.ndarray:

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
        Xp_means = self.Xs_mean[ps_area]
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
            samp_weight_ps,
        )

    def fit(
        self,
        ys: Array,
        Xs: Array,
        areas: Array,
        samp_weight: Optional[Array] = None,
        scales: Union[Array, Number] = 1,
        intercept: bool = True,
    ) -> None:
        """Fits the linear mixed models to estimate the model parameters that is the fixed
        effects, the random effects standard error and the unit level residuals' standard error. 
        In addition, the method provides statistics related to the model fitting e.g. convergence 
        status, log-likelihood, AIC, BIC, and more.  

        Args:
            ys (Array): An array of the output sample observations. 
            Xs (Array): An multi-dimensional array of the auxiliary information. 
            areas (Array): An array of the sampled area provided at the unit level. 
            samp_weight (Optional[Array], optional): An array of the sample weights. 
                Defaults to None.
            scales (Union[Array, Number], optional): The scale factor for the unit level errors. 
                If a single number of provided, the same number will be applied to all observations. Defaults to 1.
            intercept (bool, optional): An boolean to indicate whether an intercept need to be 
                added to X. Defaults to True.
        """

        areas = formats.numpy_array(areas)
        ys = formats.numpy_array(ys)
        Xs = formats.numpy_array(Xs)
        if intercept:
            if Xs.ndim == 1:
                n = Xs.shape[0]
                Xs = np.insert(Xs.reshape(n, 1), 0, 1, axis=1)
            else:
                Xs = np.insert(Xs, 0, 1, axis=1)
        if samp_weight is not None and isinstance(samp_weight, pd.DataFrame):
            samp_weight = formats.numpy_array(samp_weight)
        if isinstance(scales, (float, int)):
            scales = np.ones(ys.shape[0]) * scales
        else:
            scales = formats.numpy_array(scales)

        self.scales = scales
        self.ys = ys
        self.Xs = Xs
        self.areas = areas
        self.areas_list = np.unique(areas)

        self.afactors = dict(zip(self.areas_list, basic_functions.sumby(areas, scales)))

        reml = True if self.method == "REML" else False
        basic_model = sm.MixedLM(ys, Xs, areas)
        basic_fit = basic_model.fit(reml=reml, full_output=True,)

        self.error_std = basic_fit.scale ** 0.5
        self.fixed_effects = basic_fit.fe_params
        # self.random_effects = np.array(list(basic_fit.random_effects.values()))

        self.fe_std = basic_fit.bse_fe
        self.re_std = float(basic_fit.cov_re) ** 0.5
        self.convergence["achieved"] = basic_fit.converged
        self.convergence["iterations"] = len(basic_fit.hist[0]["allvecs"]) - 1

        nb_obs = ys.shape[0]
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

        self.ys_mean, self.Xs_mean, gamma, samp_size = area_stats(
            ys, Xs, areas, self.error_std, self.re_std, self.afactors, samp_weight
        )
        self.random_effects = gamma * (self.ys_mean - self.Xs_mean @ self.fixed_effects)
        self.gamma = dict(zip(self.areas_list, gamma))
        self.samp_size = dict(zip(self.areas_list, samp_size))

        # samp_weight = np.ones(y.size)
        if samp_weight is not None:
            beta_w = self._beta(ys, Xs, areas, samp_weight)

        self.fitted = True

    def predict(
        self, Xmean: Array, area: Array, pop_size: Optional[Array] = None, intercept: bool = True,
    ) -> None:
        """Predicts the area level means and provides the taylor MSE estimation of the estimated
        area means. 

        Args:
            Xmean (Array): a multi-dimensional array of the population means of the auxiliary   
                variables. 
            area (Array): An array of the areas in the same order as Xmean and Popsize. 
            pop_size (Optional[Array], optional): An array of the population size for the same 
                areas as in Xmean and area and in the same order. Defaults to None.
            intercept (bool, optional): [description]. Defaults to True.

        Raises:
            Exception: when predict() is called before fitting the model. 
        """

        if not self.fitted:
            raise Exception(
                "The model must be fitted first with .fit() before running the prediction."
            )

        Xmean = formats.numpy_array(Xmean)
        if intercept:
            if Xmean.ndim == 1:
                n = Xmean.shape[0]
                Xmean = np.insert(Xmean.reshape(n, 1), 0, 1, axis=1)
            else:
                Xmean = np.insert(Xmean, 0, 1, axis=1)
        self.Xbar_p = Xmean

        area = formats.numpy_array(area)
        areas_p = np.unique(area)

        ps = np.isin(area, self.areas_list)
        area_ps = area[ps]
        areas_ps = np.unique(area_ps)
        ps_area = np.isin(areas_p, areas_ps)

        if Xmean is not None:
            Xmean_ps = Xmean[ps_area]
            Xmean_pr = Xmean[~ps_area]
        else:
            Xmean_ps = Xmean_pr = None

        gamma_ps = np.asarray(list(self.gamma.values()))[ps_area]
        samp_size_ps = np.asarray(list(self.samp_size.values()))[ps_area]
        if pop_size is not None:
            pop_size = formats.numpy_array(pop_size)
            pop_size_ps = pop_size[ps_area]
            samp_rate_ps = samp_size_ps / pop_size_ps
            eta_pred = np.matmul(Xmean_ps, self.fixed_effects) + (
                samp_rate_ps + (1 - samp_rate_ps) * gamma_ps
            ) * (self.ys_mean[ps_area] - np.matmul(self.Xs_mean[ps_area], self.fixed_effects))
        elif pop_size is None:
            eta_pred = np.matmul(Xmean_ps, self.fixed_effects) + gamma_ps

        if np.sum(~ps) > 0:
            eta_r_pred = np.matmul(Xmean_pr, self.fixed_effects)
            eta_pred = np.append(eta_pred, eta_r_pred)

        self.area_est = dict(zip(areas_ps, eta_pred))

        X_ps = self.Xs[np.isin(self.areas, area)]
        A_ps = np.diag(np.zeros(Xmean.shape[1])) if Xmean.ndim >= 2 else np.asarray([0])
        for d in areas_ps:
            areadps = area_ps == d
            n_ps_d = np.sum(areadps)
            X_ps_d = X_ps[areadps]
            V_ps_d = (self.error_std ** 2) * np.diag(np.ones(n_ps_d)) + (
                self.re_std ** 2
            ) * np.ones([n_ps_d, n_ps_d])
            A_ps = A_ps + np.matmul(np.matmul(np.transpose(X_ps_d), np.linalg.inv(V_ps_d)), X_ps_d)

        a_factor_ps = np.asarray(list(self.afactors.values()))[ps_area]
        mse_ps = self._mse(
            areas_ps,
            self.Xs_mean[ps_area],
            Xmean_ps,
            gamma_ps,
            samp_size_ps,
            a_factor_ps,
            np.linalg.inv(A_ps),
        )
        self.area_mse = dict(zip(areas_ps, mse_ps))

        # TODO: add non-sampled areas prediction

    def bootstrap_mse(
        self,
        Xmean: Array,
        area: Array,
        number_reps: int = 500,
        samp_weight: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        intercept: bool = True,
    ) -> None:
        """Computes the MSE bootstrap estimates of the area level mean estimates. 

        Args:
            Xmean (Array): a multi-dimensional array of the population means of the auxiliary   
                variables. 
            area (Array): An array of the areas in the same order as Xmean. 
            number_reps (int): Number of replicates for the bootstrap method.. Defaults to 500.
            samp_weight (Optional[Array], optional): [description]. Defaults to None.
            scale (Union[Array, Number], optional): [description]. Defaults to 1.
            intercept (bool, optional): [description]. Defaults to True.

        Raises:
            Exception: when bootstrap_mse() is called before fitting the model. 
        """

        if not self.fitted:
            raise Exception(
                "The model must be fitted first with .fit() before running the prediction."
            )

        Xmean = formats.numpy_array(Xmean)
        area = formats.numpy_array(area)
        if intercept:
            if self.Xs.ndim == 1:
                n = self.Xs.shape[0]
                Xs = np.insert(self.Xs.reshape(n, 1), 0, 1, axis=1)
                Xmean = np.insert(Xmean.reshape(n, 1), 0, 1, axis=1)
            else:
                Xs = np.insert(Xs, 0, 1, axis=1)
                Xmean = np.insert(Xmean, 0, 1, axis=1)
        if samp_weight is not None and isinstance(samp_weight, pd.DataFrame):
            samp_weight = formats.numpy_array(samp_weight)

        if isinstance(scale, (float, int)):
            scale_p = np.ones(Xs.shape[0]) * scale
        else:
            scale_p = formats.numpy_array(scale)

        (
            ps,
            ps_area,
            X_ps,
            area_ps,
            areas_ps,
            Xmean_ps,
            Xmean_pr,
            xbar_ps,
            samp_weight_ps,
        ) = self._split_data(area, Xs, Xmean, samp_weight)

        samp_size_ps = np.asarray(list(self.samp_size.values()))[ps_area]
        X_ps_sorted = X_ps[np.argsort(area_ps)]
        scale_ps_ordered = scale_p[ps]
        if np.min(scale_ps_ordered) != np.max(scale_ps_ordered):
            scale_ps_ordered = scale_ps_ordered[np.argsort(area_ps)]
        nb_areas = areas_ps.shape[0]
        error = np.abs(scale_ps_ordered) * np.random.normal(
            scale=self.error_std, size=(number_reps, area_ps.shape[0])
        )
        re = np.random.normal(scale=self.re_std, size=(number_reps, nb_areas))
        mu = X_ps_sorted @ self.fixed_effects
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
            boot_fit = boot_model.fit(reml=reml,)
            boot_fe = boot_fit.fe_params
            boot_error_std = boot_fit.scale ** 0.5
            boot_re_std = float(boot_fit.cov_re) ** 0.5
            boot_ys_mean, boot_Xs_mean, boot_gamma, _ = area_stats(
                y_ps_boot[k, :],
                X_ps_sorted,
                area_ps,
                boot_error_std,
                boot_re_std,
                self.a_factor_s,
                samp_weight_ps,
            )
            boot_re = boot_gamma * (boot_ys_mean - np.matmul(boot_Xs_mean, boot_fe))
            boot_mu = np.matmul(Xmean_ps, self.fixed_effects) + re[k, :]
            boot_mu_h = np.matmul(Xmean_ps, boot_fe) + boot_re
            boot_mse[k, :] = (boot_mu_h - boot_mu) ** 2

            if k in steps:
                i += 1
                print(
                    f"\r[%-{bar_length-1}s] %d%%" % ("=" * i, 2 + (100 / bar_length) * i), end="",
                )
        print("\n")

        self.area_mse_boot = dict(zip(area_ps, np.mean(boot_mse, axis=0)))


class EbUnitModel:
    """*EbUnitModel* implements the basic Unit level model for complex indicators.

    *EbUnitModel* takes the sample data as input and fits the basic linear mixed model. 
    The user can pick between restricted maximum likelihood (REML) or maximum likelihood (ML) 
    to fit the model parameters. Also, EbUnitModel predicts the areas means and provides 
    the point and mean squared error (MSE) estimates of the empirical Bayes linear 
    unbiased (EBLUP). User can also obtain the bootstrap mse estimates of the MSE.

    *EbUnitModel* requires the user to provide the indicator function. The indicator function is 
    expected to take the array of output sample observations as input and possibly some additional 
    parameters needed to compute the indicator. The indicator function outputs an aggregated value.
    For example, the poverty gap indicator can have the following signature 
    pov_gap(y: array, pov_line: float) -> float. If the indicator function different outputs by 
    area then the self.area_list can be used to incorporate different logics across areas.

    Also, *EbUnitModel* can use Boxcox to transform the output sample values in order to reduce 
    the asymmetry in the datawhen fitting the linear mixed model. 

    Setting attributes
        | method (str): the fitting method of the model parameters which can take the possible 
        |   values restricted maximum likelihood (REML) or maximum likelihood (ML). 
        |   If not specified, "REML" is used as default.  
        | indicator (function): a user defined function to compute the indicator. 
        | boxcox (dict): contains the *lambda* parameter of the Boxcox and a constant for the 
        | log-transformation of the Boxcox.  
    
    Sample related attributes
        | ys (array): the output sample observations. 
        | Xs (ndarray): the auxiliary information. 
        | scales (array): an array of scaling parameters for the unit levels errors.
        | afactors (array): sum of the inverse squared of scale.
        | areas (array): the full vector of small areas from the sampled observations.  
        | areas_list (array): the list of small areas from the sample data.
        | samp_size (dict): the sample size per small areas from the sample. 
        | ys_mean (array): sample area means of the output variable. 
        | Xs_mean (ndarray): sample area means of the auxiliary variables.

    Model fitting attributes
        | fitted (boolean): indicates whether the model has been fitted or not. 
        | fixed_effects (array): the estimated fixed effects of the regression model. 
        | fe_std (array): the estimated standard errors of the fixed effects. 
        | random_effects (array): the estimated area level random effects.
        |   associated with the small areas. 
        | re_std (number): the estimated standard error of the random effects. 
        | error_std (number): standard error of the unit level residuals. 
        | convergence (dict): a dictionnary holding the convergence status and the number of 
        |   iterations from the model fitting algorithm. 
        | goodness (dict): a dictionary holding the log-likelihood, AIC, and BIC.
        | gamma (dict): ratio of the between-area variability (re_std**2) to the total 
        |   variability (re_std**2 + error_std**2 / a_factor). 

    Prediction related attributes
        | areap (array): the list of areas for the prediction. 
        | number_reps (int): number of replicates for the bootstrap MSE estimation. 
        | area_est (array): area level EBLUP estimates. 
        | area_mse (array): area level taylor estimation of the MSE. 
        | area_mse_boot (array): area level bootstrap estimation of the MSE.

    Main methods
        | fit(): fits the linear mixed model to estimate the model parameters using REMl or ML
        |   methods. 
        | predict(): predicts the area level indicator estimates which includes both the point 
        |   estimates and the taylor MSE estimate. 
        | bootstrap_mse(): computes the area level bootstrap MSE estimates of the indicator.
    """

    def __init__(
        self,
        method: str = "REML",
        boxcox: Optional[float] = None,
        constant: Number = 0,
        indicator: Optional[Any] = None,
    ):

        # Setting
        self.method: str = method.upper()
        if self.method not in ("REML", "ML"):
            raise AssertionError("Value provided for method is not valid!")
        self.indicator = indicator
        self.number_samples: Optional[int] = None
        self.boxcox = {"lambda": boxcox, "constant": constant}

        # Sample data
        self.scales: np.ndarray = np.array([])
        self.afactors: Dict[Any, float] = {}
        self.ys: np.ndarray = np.array([])
        self.Xs: np.ndarray = np.array([])
        self.areas: np.ndarray = np.array([])
        self.areas_list: np.ndarray = np.array([])
        self.samp_size: Dict[Any, int] = {}
        self.ys_mean: np.ndarray = np.array([])
        self.Xs_mean: np.ndarray = np.array([])

        # Fitted data
        self.fitted: bool = False
        self.fixed_effects: np.ndarray = np.array([])
        self.fe_std: np.ndarray = np.array([])
        self.random_effects: np.ndarray = np.array([])
        self.re_std: float = 0
        self.error_std: float = 0
        self.convergence: Dict[str, Union[float, int, bool]] = {}
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC
        self.gamma: Dict[Any, float] = {}

        # Predict(ion/ed) data
        self.number_reps: int = 0
        self.area_est: Dict[Any, float] = {}
        self.area_mse: Dict[Any, float] = {}
        self.area_mse_boot: Dit[Any, float] = {}

    def _transformation(self, y: np.ndarray, inverse: bool) -> np.ndarray:
        if self.boxcox["lambda"] is None:
            z = y
        elif self.boxcox["lambda"] == 0.0:
            if inverse:
                z = np.exp(y) - self.boxcox["constant"]
            else:
                z = np.log(y + self.boxcox["constant"])
        elif self.boxcox["lambda"] != 0.0:
            if inverse:
                z = np.exp(np.log(1 + y * self.boxcox["lambda"]) / self.boxcox["lambda"])
            else:
                z = np.power(y, self.boxcox["lambda"]) / self.boxcox["lambda"]
        return z

    def fit(
        self,
        ys: Array,
        Xs: Array,
        areas: Array,
        samp_weight: Optional[Array] = None,
        scales: Union[Array, Number] = 1,
        intercept: bool = True,
    ) -> None:
        """Fits the linear mixed models to estimate the model parameters that is the fixed
        effects, the random effects standard error and the unit level residuals' standard error. 
        In addition, the method provides statistics related to the model fitting e.g. convergence 
        status, log-likelihood, AIC, BIC, and more.

        Args:
            ys (Array): An array of the output sample observations. 
            Xs (Array): An multi-dimensional array of the sample auxiliary information. 
            areas (Array): provides the area of the sampled observations. 
            samp_weight (Optional[Array], optional): An array of the sample weights. 
                Defaults to None.
            scales (Union[Array, Number], optional): the scale factor for the unit level errors. 
                If a single number of provided, the same number will be applied to all observations. Defaults to 1.
            intercept (bool, optional): An boolean to indicate whether an intercept need to be 
                added to Xs. Defaults to True
        """

        ys_transformed = basic_functions.transform(
            ys, llambda=self.boxcox["lambda"], constant=self.boxcox["constant"], inverse=False
        )

        eblup_ul = EblupUnitModel()
        eblup_ul.fit(
            ys_transformed, Xs, areas, samp_weight, scales, intercept,
        )

        self.scales = eblup_ul.scales
        self.ys = eblup_ul.ys
        self.Xs = eblup_ul.Xs
        self.areas = eblup_ul.areas
        self.areas_list = eblup_ul.areas_list
        self.afactors = eblup_ul.afactors
        self.error_std = eblup_ul.error_std
        self.fixed_effects = eblup_ul.fixed_effects
        self.fe_std = eblup_ul.fe_std
        self.re_std = eblup_ul.re_std
        self.convergence = eblup_ul.convergence
        self.goodness = eblup_ul.goodness
        self.ys_mean = eblup_ul.ys_mean
        self.Xs_mean = eblup_ul.Xs_mean
        self.gamma = eblup_ul.gamma
        self.samp_size = eblup_ul.samp_size
        self.fitted = eblup_ul.fitted

    def _predict_indicator(
        self,
        number_samples: int,
        y_s: np.ndarray,
        X_s: np.ndarray,
        area_s: np.ndarray,
        X_r: np.ndarray,
        area_r: np.ndarray,
        arear_list: np.ndarray,
        fixed_effects: np.ndarray,
        gamma: np.ndarray,
        sigma2e: float,
        sigma2u: float,
        scale: np.ndarray,
        max_array_length: int,
        indicator: Callable[..., np.ndarray],
        show_progress: bool,
        *args: Any,
    ) -> np.ndarray:
        nb_arear = len(arear_list)
        mu_r = X_r @ fixed_effects

        if show_progress:
            k = 0
            bar_length = min(50, nb_arear)
            steps = np.linspace(1, nb_arear - 1, bar_length).astype(int)
            print(f"Generating the {number_samples} replicates samples")

        eta = np.zeros((number_samples, nb_arear)) * np.nan
        for i, d in enumerate(arear_list):
            # print(d)
            oos = area_r == d
            mu_dr = mu_r[oos]
            ss = self.areas_s == d
            ybar_d = self.ys_mean[ss]
            xbar_d = self.Xs_mean[ss]
            mu_bias_dr = self.gamma[d] * (ybar_d - xbar_d @ fixed_effects)
            scale_dr = scale[oos]
            N_dr = np.sum(oos)
            cycle_size = max(int(max_array_length // N_dr), 1)
            number_cycles = int(number_samples // cycle_size)
            last_cycle_size = number_samples % cycle_size

            for j in range(number_cycles + 1):
                if j == number_cycles:
                    cycle_size = last_cycle_size
                re_effects = np.random.normal(
                    scale=(sigma2u * (1 - self.gamma[d])) ** 0.5, size=cycle_size
                )
                errors = np.random.normal(
                    scale=scale_dr * (sigma2e ** 0.5), size=(cycle_size, N_dr)
                )
                y_dr_j = mu_dr[None, :] + mu_bias_dr + re_effects[:, None] + errors
                if j == 0:
                    y_dr = y_dr_j
                else:
                    y_dr = np.append(y_dr, y_dr_j, axis=0)

            if show_progress:
                if i in steps:
                    k += 1
                    print(
                        f"\r[%-{bar_length}s] %d%%"
                        % ("=" * (k + 1), (k + 1) * (100 / bar_length)),
                        end="",
                    )

            y_d = np.append(y_dr, np.tile(y_s[area_s == d], [number_samples, 1]), axis=1)
            z_d = basic_functions.transform(
                y_d, llambda=self.boxcox["lambda"], constant=self.boxcox["constant"], inverse=True
            )
            eta[:, i] = np.apply_along_axis(indicator, axis=1, arr=z_d, *args)  # *)

        if show_progress:
            print("\n")

        return np.mean(eta, axis=0)

    def predict(
        self,
        Xr: Array,
        arear: Array,
        indicator: Callable[..., Array],
        number_samples: int,
        scaler: Union[Array, Number] = 1,
        intercept: bool = True,
        max_array_length: int = int(100e6),
        show_progress: bool = True,
        *args: Any,
    ) -> None:
        """Predicts the area level means and provides the taylor MSE estimation of the estimated
        area means. 

        Args:
            Xr (Array): an multi-dimensional array of the out of sample auxiliary variables.
            arear (Array): provides the area of the out of sample units.
            indicator (Callable[..., Array]): a user defined function which computes the area level
                indicators. The function should take y (output variable) as the first parameters, 
                additional parameters can be used. Use *args* to transfer the additional 
                parameters.
            number_samples (int): number of replicates for the Monte-Carlo (MC) algorithm.  
            scaler (Union[Array, Number], optional): the scale factor for the unit level errors. 
                If a single number of provided, the same number will be applied to all observations. Defaults to 1.
            intercept (bool, optional): An boolean to indicate whether an intercept need to be 
                added to Xr. Defaults to True.
            max_array_length (int, optional): controls the number of replicates to generate at 
                the same time. This parameter helps with performance. The number can be reduce or 
                increase based on the user's computer RAM capacity. Defaults to int(100e6).
            show_progress (bool, optional): shows a bar progress of the MC replicates 
                calculations. Defaults to True.

        Raises:
            Exception: when predict() is called before fitting the model.
        """

        if not self.fitted:
            raise Exception(
                "The model must be fitted first with .fit() before running the prediction."
            )

        self.number_samples = int(number_samples)

        if isinstance(scaler, (float, int)):
            scaler = np.ones(Xr.shape[0]) * scaler
        else:
            scale = formats.numpy_array(scaler)
        area = formats.numpy_array(arear)
        self.arear_list = np.unique(arear)
        Xr = formats.numpy_array(Xr)
        if intercept:
            if Xr.ndim == 1:
                n = Xr.shape[0]
                Xr = np.insert(Xr.reshape(n, 1), 0, 1, axis=1)
            else:
                Xr = np.insert(Xr, 0, 1, axis=1)

        area_est = self._predict_indicator(
            self.number_samples,
            self.ys,
            self.Xs,
            self.areas,
            Xr,
            arear,
            self.arear_list,
            self.fixed_effects,
            self.gamma,
            self.error_std ** 2,
            self.re_std ** 2,
            scaler,
            max_array_length,
            indicator,
            show_progress,
            *args,
        )

        self.area_est = dict(zip(self.arear_list, area_est))

    def bootstrap_mse(
        self,
        number_reps: int,
        indicator: Callable[..., Array],
        Xr: Array,
        arear: Array,
        scaler: Union[Array, Number] = 1,
        intercept: bool = True,
        max_array_length: int = int(100e6),
        *args: Any,
    ) -> None:
        """Computes the MSE bootstrap estimates of the area level indicator estimates.

        Args:
            Xr (Array): an multi-dimensional array of the out of sample auxiliary variables.
            arear (Array): provides the area of the out of sample units.
            indicator (Callable[..., Array]): [description]
            number_reps (int): [description]
            scaler (Union[Array, Number], optional): [description]. Defaults to 1.
            intercept (bool, optional): [description]. Defaults to True.
            max_array_length (int, optional): [description]. Defaults to int(100e6).
        """

        X_r = formats.numpy_array(Xr)
        area_r = formats.numpy_array(arear)
        arear_list = np.unique(area_r)

        if intercept:
            if X_r.ndim == 1:
                n = X_r.shape[0]
                X_r = np.insert(X_r.reshape(n, 1), 0, 1, axis=1)
            else:
                X_r = np.insert(X_r, 0, 1, axis=1)

        if isinstance(scale, (float, int)):
            scale_r = np.ones(X_r.shape[0]) * scale
        else:
            scale_r = formats.numpy_array(scale)

        ps = np.isin(area_r, self.areas_list)
        areas_ps = np.unique(area_r[ps])
        nb_areas_ps = areas_ps.size
        area_s = self.areas[np.isin(self.areas, arear_list)]
        area = np.append(area_r, area_s)
        scale_s = self.scales[np.isin(self.areas, arear_list)]
        scale = np.append(scale_r, scale_s)
        _, N_d = np.unique(area, return_counts=True)
        X_s = self.Xs[np.isin(self.areas, arear_list)]
        X = np.append(X_r, X_s, axis=0)

        aboot_factor = np.zeros(nb_areas_ps)

        indice_dict = {}
        area_dict = {}
        scale_dict = {}
        scale_s_dict = {}
        a_factor_dict = {}
        sample_size_dict = {}
        sample_dict = {}
        X_dict = {}
        X_s_dict = {}
        for i, d in enumerate(arear_list):
            area_ds = area_s == d
            indice_dict[d] = area == d
            area_dict[d] = area[indice_dict[d]]
            scale_dict[d] = scale[indice_dict[d]]
            a_factor_dict[d] = self.a_factor_s[d] if d in self.areas_s else 0
            sample_size_dict[d] = self.samp_size[d] if d in self.areas_s else 0
            scale_s_dict[d] = scale_s[area_s == d] if d in self.areas_s else 0
            X_dict[d] = X[indice_dict[d]]
            X_s_dict[d] = X_s[area_s == d]  # X_dict[d][sample_dict[d]]

        cycle_size = max(int(max_array_length // sum(N_d)), 1)
        number_cycles = int(number_reps // cycle_size)
        last_cycle_size = number_reps % cycle_size
        number_cycles = number_cycles + 1 if last_cycle_size > 0 else number_cycles

        k = 0
        bar_length = min(50, number_cycles * nb_areas_ps)
        steps = np.linspace(1, number_cycles * nb_areas_ps - 1, bar_length).astype(int)

        eta_pop_boot = np.zeros((number_reps, nb_areas_ps))
        eta_samp_boot = np.zeros((number_reps, nb_areas_ps))
        y_samp_boot = np.zeros((number_reps, np.sum(list(sample_size_dict.values()))))
        print(f"Generating the {number_reps} bootstrap replicate populations")
        for b in range(number_cycles):
            start = b * cycle_size
            end = (b + 1) * cycle_size
            if b == number_cycles - 1:
                end = number_reps
                cycle_size = last_cycle_size

            for i, d in enumerate(areas_ps):
                aboot_factor[i] = a_factor_dict[d]

                re_d = np.random.normal(
                    scale=self.re_std * (1 - self.gamma[d]) ** 0.5, size=cycle_size
                )
                err_d = np.random.normal(
                    scale=self.error_std * scale_dict[d], size=(cycle_size, np.sum(indice_dict[d]))
                )
                yboot_d = (X_dict[d] @ self.fixed_effects)[None, :] + re_d[:, None] + err_d
                zboot_d = basic_functions.transform(
                    yboot_d,
                    llambda=self.boxcox["lambda"],
                    constant=self.boxcox["constant"],
                    inverse=True,
                )
                eta_pop_boot[start:end, i] = indicator(zboot_d, *args)

                if i == 0:
                    yboot_s = yboot_d[:, -sample_size_dict[d] :]
                else:
                    yboot_s = np.append(yboot_s, yboot_d[:, -sample_size_dict[d] :], axis=1)

                run_id = b * nb_areas_ps + i
                if run_id in steps:
                    k += 1
                    print(
                        f"\r[%-{bar_length}s] %d%%"
                        % ("=" * (k + 1), (k + 1) * (100 / bar_length)),
                        end="",
                    )

            y_samp_boot[start:end, :] = yboot_s

        print("\n")

        k = 0
        bar_length = min(50, number_reps)
        steps = np.linspace(1, number_reps, bar_length).astype(int)

        reml = True if self.method == "REML" else False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            beta_ols = sm.OLS(y_samp_boot[0, :], X_s).fit().params
        resid_ols = y_samp_boot[0, :] - np.matmul(X_s, beta_ols)
        re_ols = basic_functions.sumby(area_s, resid_ols) / basic_functions.sumby(
            area_s, np.ones(area_s.size)
        )
        print(f"Fitting and predicting using each of the {number_reps} bootstrap populations")
        for b in range(number_reps):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                boot_model = sm.MixedLM(y_samp_boot[b, :], X_s, area_s)
                boot_fit = boot_model.fit(
                    reml=reml,
                    start_params=np.append(beta_ols, np.std(re_ols) ** 2),
                    full_output=True,
                )

            gammaboot = float(boot_fit.cov_re) / (
                float(boot_fit.cov_re) + boot_fit.scale * (1 / aboot_factor)
            )

            eta_samp_boot[b, :] = self._predict_indicator(
                self.number_samples,
                y_samp_boot[b, :],
                X_s,
                area_s,
                X_r,
                area_r,
                np.unique(area_r),
                boot_fit.fe_params,
                gammaboot,
                boot_fit.scale,
                float(boot_fit.cov_re),
                scale_r,
                max_array_length,
                indicator,
                False,
                *args,
            )

            if b in steps:
                k += 1
                print(
                    f"\r[%-{bar_length}s] %d%%" % ("=" * (k + 1), (k + 1) * (100 / bar_length)),
                    end="",
                )
        print("\n")

        mse_boot = np.mean(np.power(eta_samp_boot - eta_pop_boot, 2), axis=0)
        self.area_mse_boot = dict(zip(self.arear_list, area_mse_boot))


class EllUnitModel:
    """*EllUnitModel* implements the basic Unit level model for complex indicators.

    *EllUnitModel* takes the sample data as input and fits the basic linear mixed model. 
    The user can pick between restricted maximum likelihood (REML), maximum likelihood (ML), 
    and method of moments (MOM) to fit the model parameters. Also, EllUnitModel predicts the areas means and provides the point and mean squared error (MSE) estimates of the empirical Bayes 
    linear unbiased (EBLUP). 

    *EllUnitModel* requires the user to provide the indicator function. The indicator function is 
    expected to take the array of output sample observations as input and possibly some additional 
    parameters needed to compute the indicator. The indicator function outputs an aggregated value.
    For example, the poverty gap indicator can have the following signature 
    pov_gap(y: array, pov_line: float) -> float. If the indicator function different outputs by 
    area then the self.area_list can be used to incorporate different logics across areas.

    Also, EllUnitModel can use Boxcox to transform the output sample values in order to reduce the 
    asymmetry in the datawhen fitting the linear mixed model. 

    Setting attributes
        | method (str): the fitting method of the model parameters which can take the possible 
        |   values restricted maximum likelihood (REML),  maximum likelihood (ML), and method of 
        |   moments (MOM). If not specified, "MOM" is used as default.  
        | indicator (function): a user defined function to compute the indicator. 
        | boxcox (dict): contains the *lambda* parameter of the Boxcox and a constant for the 
        |   log-transformation of the Boxcox.  
    
    Sample related attributes
        | ys (array): the output sample observations. 
        | Xs (ndarray): the auxiliary information. 
        | scales (array): an array of scaling parameters for the unit levels errors.
        | afactors (array): sum of the inverse squared of scale.
        | areas (array): the full vector of small areas from the sampled observations.  
        | areas_list (array): the list of small areas from the sample data.
        | samp_size (dict): the sample size per small areas from the sample. 
        | ys_mean (array): sample area means of the output variable. 
        | Xs_mean (ndarray): sample area means of the auxiliary variables.

    Model fitting attributes
        | fitted (boolean): indicates whether the model has been fitted or not. 
        | fixed_effects (array): the estimated fixed effects of the regression model. 
        | fe_std (array): the estimated standard errors of the fixed effects. 
        | random_effects (array): the estimated area level random effects
        | re_std (number): the estimated standard error of the random effects. 
        | error_std (number): the estimated standard error of the unit level residuals. 
        | convergence (dict): a dictionnary holding the convergence status and the number of 
        |   iterations from the model fitting algorithm. 
        | goodness (dict): a dictionary holding the log-likelihood, AIC, and BIC.
        | gamma (dict): ratio of the between-area variability (re_std**2) to the total 
        |   variability (re_std**2 + error_std**2 / a_factor). 

    Prediction related attributes
        | areap (array): the list of areas for the prediction. 
        | number_reps (int): number of replicates for the bootstrap MSE estimation. 
        | area_est (array): area level EBLUP estimates. 
        | area_mse (array): area level taylor estimation of the MSE. 

    Main methods
        | fit(): fits the linear mixed model to estimate the model parameters using REMl or ML
        |   methods. 
        | predict(): predicts the area level indicator estimates which includes both the point 
        |   estimates and the taylor MSE estimate. 
    """

    def __init__(
        self,
        method: str = "MOM",
        boxcox: Optional[float] = None,
        constant: Number = 0,
        indicator: Optional[Any] = None,
    ):

        # Setting
        self.method: str = method.upper()
        if self.method not in ("REML", "ML", "MOM"):
            raise AssertionError("Value provided for method is not valid!")
        self.indicator = indicator
        self.boxcox = {"lambda": boxcox, "constant": constant}

        # Sample data
        self.scales: np.ndarray = np.array([])
        self.afactors: Dict[Any, float] = {}
        self.ys: np.ndarray = np.array([])
        self.Xs: np.ndarray = np.array([])
        self.areas: np.ndarray = np.array([])
        self.areas_list: np.ndarray = np.array([])
        self.samp_size: Dict[Any, int] = {}
        self.ys_mean: np.ndarray = np.array([])
        self.Xs_mean: np.ndarray = np.array([])

        # Fitted data
        self.fitted: bool = False
        self.fixed_effects: np.ndarray = np.array([])
        self.fe_std: np.ndarray = np.array([])
        self.random_effects: np.ndarray = np.array([])
        self.re_std: float = 0
        self.error_std: float = 0
        self.convergence: Dict[str, Union[float, int, bool]] = {}
        self.goodness: Dict[str, float] = {}  # loglikehood, deviance, AIC, BIC
        self.gamma: Dict[Any, float] = {}

        # Predict(ion/ed) data
        self.areap: np.ndarray = np.array([])
        self.pop_size: Dict[Any, float] = {}
        self.Xp_mean: np.ndarray = np.array([])
        self.number_reps: int = 0
        self.area_est: Dict[Any, float] = {}
        self.area_mse: Dict[Any, float] = {}

    def fit(
        self,
        ys: Array,
        Xs: Array,
        areas: Array,
        samp_weight: Optional[Array] = None,
        scales: Union[Array, Number] = 1,
        intercept: bool = True,
    ) -> None:
        """Fits the linear mixed models to estimate the model parameters that is the fixed
        effects, the random effects standard error and the unit level residuals' standard error.
        In addition, the method provides statistics related to the model fitting e.g. convergence 
        status, log-likelihood, AIC, BIC, and more.

        Args:
            ys (Array):  an array of the output sample observations. 
            Xs (Array): an multi-dimensional array of the sample auxiliary information.
            areas (Array): provides the area of the sampled observations. 
            samp_weight (Optional[Array], optional): An array of the sample weights. 
                Defaults to None.
            scales (Union[Array, Number], optional): the scale factor for the unit level errors. 
                If a single number of provided, the same number will be applied to all observations. Defaults to 1.
            intercept (bool, optional): An boolean to indicate whether an intercept need to be 
                added to Xs. Defaults to True.
        """

        ys = formats.numpy_array(ys)
        Xs = formats.numpy_array(Xs)
        if intercept:
            if Xs.ndim == 1:
                n = Xs.shape[0]
                Xs = np.insert(Xs.reshape(n, 1), 0, 1, axis=1)
            else:
                Xs = np.insert(Xs, 0, 1, axis=1)
        if samp_weight is not None and isinstance(samp_weight, pd.DataFrame):
            samp_weight = formats.numpy_array(samp_weight)
        if isinstance(scale, (float, int)):
            scales = np.ones(ys.shape[0]) * scales
        else:
            scales = formats.numpy_array(scales)

        if self.method in ("REML", "ML"):
            eb_ul = EbUnitModel(
                method=self.method, boxcox=self.boxcox["lambda"], constant=self.boxcox["constant"]
            )
            eb_ul.fit(
                ys, Xs, areas, samp_weight, scales, False,
            )
            self.scales = eb_ul.scales
            self.afactors = eb_ul.afactors
            self.ys = eb_ul.ys
            self.Xs = eb_ul.Xs
            self.areas = eb_ul.areas
            self.areas_list = eb_ul.areas_list
            self.error_std = eb_ul.error_std
            self.fixed_effects = eb_ul.fixed_effects
            self.fe_std = eb_ul.fe_std
            self.re_std = eb_ul.re_std
            self.convergence = eb_ul.convergence
            self.goodness = eb_ul.goodness
            self.ys_mean = eb_ul.ys_mean
            self.Xs_mean = eb_ul.Xs_mean
            self.gamma = eb_ul.gamma
            self.samp_size = eb_ul.samp_size
            self.fitted = eb_ul.fitted
        else:
            eb_ul = EbUnitModel(boxcox=self.boxcox["lambda"], constant=self.boxcox["constant"])
            ols_fit = sm.OLS(ys, Xs).fit()
            beta_ols = ols_fit.params
            resid_ols = ys - np.matmul(Xs, beta_ols)
            re_ols = basic_functions.sumby(self.area_s, resid_ols) / basic_functions.sumby(
                self.areas, np.ones(self.areas.size)
            )
            self.error_std = 111
            self.fixed_effects = beta_ols
            self.scales = scales
            self.ys = ys
            self.Xs = Xs
            self.areas = areas
            self.areas_list = np.unique(areas)
            self.afactors = dict(zip(self.areas_list, basic_functions.sumby(areas, scales)))
            self.ys_mean, self.Xs_mean, _, samp_size = area_stats(
                ys, Xs, area, 0, 1, self.afactors, samp_weight
            )
            self.samp_size = dict(zip(self.areas_list, samp_size))
            # self.fe_std = eblupUL.fe_std
            # self.re_std = eblupUL.re_std
            self.fitted = True

    def _predict_indicator_parametric(
        self,
        mu: Array,
        area: Array,
        sigma2u: Number,
        sigma2e: Number,
        indicator: Callable[..., Array],
        number_samples: int,
        scale: Array,
        max_array_length: int,
        show_progress: bool,
        *args: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:

        areas = np.unique(area)
        nb_areas = len(areas)
        if show_progress:
            k = 0
            bar_length = min(50, nb_areas)
            steps = np.linspace(1, nb_areas - 1, bar_length).astype(int)
            print(f"Generating the {number_samples} replicates samples")

        eta = np.zeros((number_samples, nb_areas)) * np.nan
        for i, d in enumerate(areas):
            aread = area == d
            mu_d = mu[aread]
            scale_d = scale[aread]
            N_d = np.sum(aread)
            cycle_size = max(int(max_array_length // N_d), 1)
            number_cycles = int(number_samples // cycle_size)
            last_cycle_size = number_samples % cycle_size

            for j in range(number_cycles + 1):
                if j == number_cycles:
                    cycle_size = last_cycle_size
                re_effects = np.random.normal(scale=sigma2u ** 0.5, size=cycle_size)
                errors = np.random.normal(scale=scale_d * (sigma2e ** 0.5), size=(cycle_size, N_d))
                y_d_j = mu_d[None, :] + re_effects[:, None] + errors
                if j == 0:
                    y_d = y_d_j
                else:
                    y_d = np.append(y_d, y_d_j, axis=0)

            if show_progress:
                if i in steps:
                    k += 1
                    print(
                        f"\r[%-{bar_length}s] %d%%"
                        % ("=" * (k + 1), (k + 1) * (100 / bar_length)),
                        end="",
                    )

            z_d = basic_functions.transform(
                y_d, llambda=self.boxcox["lambda"], constant=self.boxcox["constant"], inverse=True
            )
            eta[:, i] = np.apply_along_axis(indicator, axis=1, arr=z_d, *args)

        if show_progress:
            print("\n")

        ell_estimate = np.mean(eta, axis=0)
        ell_mse = np.mean(np.power(eta - ell_estimate[None, :], 2), axis=0)

        return ell_estimate, ell_mse

    def _predict_indicator_nonparametric(
        self,
        number_samples: int,
        indicator: Callable[..., np.ndarray],
        mu: np.ndarray,
        area: np.ndarray,
        total_residuals: np.ndarray,
        max_array_length: int,
        show_progress: bool,
        *args: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:

        areas = np.unique(area)
        nb_areas = len(areas)
        if show_progress:
            k = 0
            bar_length = min(50, nb_areas)
            steps = np.linspace(1, nb_areas - 1, bar_length).astype(int)
            print(f"Generating the {number_samples} replicates samples")

        area_effects = basic_functions.averageby(self.area_s, total_residuals)
        for i, d in enumerate(self.areas_s):
            total_residuals_d = total_residuals[self.area_s == d]
            if i == 0:
                unit_errors = total_residuals_d - area_effects[i]
            else:
                unit_errors = np.append(unit_errors, total_residuals_d - area_effects[i])

        eta = np.zeros((number_samples, nb_areas)) * np.nan
        for i, d in enumerate(areas):
            aread = area == d
            mu_d = mu[aread]
            N_d = np.sum(aread)
            cycle_size = max(int(max_array_length // N_d), 1)
            number_cycles = int(number_samples // cycle_size)
            last_cycle_size = number_samples % cycle_size

            for j in range(number_cycles + 1):
                if j == number_cycles:
                    cycle_size = last_cycle_size
                re_effects = np.random.choice(area_effects, size=cycle_size)
                errors = np.random.choice(unit_errors, size=(cycle_size, N_d))
                y_d_j = mu_d[None, :] + re_effects[:, None] + errors
                if j == 0:
                    y_d = y_d_j
                else:
                    y_d = np.append(y_d, y_d_j, axis=0)

            if show_progress:
                if i in steps:
                    k += 1
                    print(
                        f"\r[%-{bar_length}s] %d%%"
                        % ("=" * (k + 1), (k + 1) * (100 / bar_length)),
                        end="",
                    )

        if show_progress:
            print("\n")

        ell_estimate = np.mean(eta, axis=0)
        ell_mse = np.mean(np.power(eta - ell_estimate[None, :], 2), axis=0)

        return ell_estimate, ell_mse

    def predict(
        self,
        X: Array,
        area: Array,
        indicator: Callable[..., Array],
        number_samples: int,
        scale: Array = 1,
        intercept: bool = True,
        max_array_length: int = int(100e6),
        show_progress: bool = True,
        *args: Any,
    ) -> None:
        """Predicts the area level indicator and its the MSE estimates. 

        Args:
            X (Array): an multi-dimensional array of the auxiliary variables for the population.
            area (Array): provides the area of the population units.
            indicator (Callable[..., Array]): a user defined function which computes the area level
                indicators. The function should take y (output variable) as the first parameters, 
                additional parameters can be used. Use *args* to transfer the additional 
                parameters.
            number_samples (int): [description]
            scale (Array, optional): [description]. Defaults to 1.
            intercept (bool, optional): An boolean to indicate whether an intercept need to be 
                added to X. Defaults to True.
            max_array_length (int, optional): controls the number of replicates to generate at 
                the same time. This parameter helps with performance. The number can be reduce or 
                increase based on the user's computer RAM capacity. Defaults to int(100e6).
            show_progress (bool, optional): shows a bar progress of the MC replicates 
                calculations. Defaults to True.

        Raises:
            Exception: when predict() is called before fitting the model.
        """

        if not self.fitted:
            raise Exception(
                "The model must be fitted first with .fit() before running the prediction."
            )

        self.number_samples = int(number_samples)
        if isinstance(scale, (float, int)):
            scale = np.ones(X.shape[0]) * scale
        else:
            scale = formats.numpy_array(scale)
        area = formats.numpy_array(area)
        self.areas_p = np.unique(area)
        X = formats.numpy_array(X)
        if intercept:
            if X.ndim == 1:
                n = X.shape[0]
                X = np.insert(X.reshape(n, 1), 0, 1, axis=1)
            else:
                X = np.insert(X, 0, 1, axis=1)
            # (
            #     ps,
            #     ps_area,
            #     X_ps,
            #     area_ps,
            #     areas_ps,
            #     _,
            #     _,
            #     Xp_means,
            #     a_factor_ps,
            #     samp_size_ps,
            #     gamma_ps,
            #     samp_weight_ps,
            # ) = EblupUnitModel._split_data(area, X, None, samp_weight)
        mu = X @ self.fixed_effects

        if self.method in ("REML", "ML"):
            area_est, area_mse = self._predict_indicator_parametric(
                self.number_samples,
                indicator,
                mu,
                area,
                self.re_std ** 2,
                self.error_std ** 2,
                scale,
                max_array_length,
                show_progress,
                *args,
            )
        elif self.method in ("MOM"):
            # y_transformed_s = basic_functions.transform(
            #    self.y_s, llambda=self.boxcox["lambda"], inverse=False
            # )
            # total_residuals = y_transformed_s - self.X_s @ self.fixed_effects
            total_residuals = self.y_s - self.X_s @ self.fixed_effects
            area_est, area_mse = self._predict_indicator_nonparametric(
                self.number_samples,
                indicator,
                mu,
                area,
                total_residuals,
                max_array_length,
                show_progress,
                *args,
            )

        self.area_est = dict(zip(self.areas_p, area_est))
        self.area_mse = dict(zip(self.areas_p, area_mse))
