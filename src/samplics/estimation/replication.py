import math

from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from numpy.lib.arraysetops import isin
from scipy.stats import t as student

from samplics.estimation.expansion import _SurveyEstimator
from samplics.utils import formats
from samplics.utils.types import Array, DictStrNum, Number, StringNumber


TypeRepEst = TypeVar("TypeRepEst", bound="ReplicateEstimator")


class ReplicateEstimator(_SurveyEstimator):
    """*ReplicateEstimator* implements replicate-based variance approximations.


    Attributes
        | point_est (dict): point estimate of the parameter of interest.
        | variance (dict): variance estimate of the parameter of interest.
        | stderror (dict): standard error of the parameter of interest.
        | coef_var (dict): estimate of the coefficient of variation.
        | deff (dict): estimate of the design effect due to weighting.
        | lower_ci (dict): estimate of the lower bound of the confidence interval.
        | upper_ci (dict): estimate of the upper bound of the confidence interval.
        | degree_of_freedom (int): degree of freedom for the confidence interval.
        | alpha (float): significant level for the confidence interval
        | strata (list): list of the strata in the sample.
        | domains (list): list of the domains in the sample.
        | method (str): variance estimation method.
        | parameter (str): the parameter of the population to estimate e.g. total.
        | number_strata (int): number of strata.
        | number_psus (int): number of primary sampling units (psus).
        | conservative (bool): indicate whether to produce conservative variance estimates.
        | number_reps (int): number of replicate weights.
        | rep_coefs (array): coefficients associated to the replicate weights.
        | fay_coef (float): Fay coefficient for the the BRR-Fay algorithm.
        | random_seed (int): random seed for reproducibility.

    Methods
        | estimate(): produces the point estimate of the parameter of interest with the associated
        |   measures of precision.
    """

    def __init__(
        self,
        method: str,
        parameter: str,
        rep_weight_cls: Optional[Any] = None,
        fay_coef: Optional[Number] = None,
        alpha: float = 0.05,
        random_seed: Optional[int] = None,
    ) -> None:

        if method.lower() not in ("bootstrap", "brr", "jackknife"):
            raise ValueError("method must be 'bootstrap', 'brr', or 'jackknife'")

        super().__init__(parameter, alpha, random_seed)
        self.method = method.lower()
        self.conservative = False
        self.degree_of_freedom: Optional[int] = None
        self.number_reps: Optional[int] = None
        self.rep_coefs: Optional[Union[np.ndarray, Number]] = None
        if method == "brr" and fay_coef is not None:
            self.fay_coef = fay_coef
        elif method == "brr":
            self.fay_coef = 0
        if rep_weight_cls is not None:
            self.number_reps = rep_weight_cls.number_reps
            self.rep_coefs = rep_weight_cls.rep_coefs
            self.degree_of_freedom = rep_weight_cls.degree_of_freedom
            self.fay_coef = rep_weight_cls.fay_coef if self.method == "brr" else None

    def _rep_point(
        self, y: np.ndarray, rep_weights: np.ndarray, x: Optional[np.ndarray]
    ) -> np.ndarray:
        if self.parameter in ("proportion", "mean"):
            return np.asarray(
                np.sum(rep_weights * y[:, None], axis=0) / np.sum(rep_weights, axis=0)
            )
        elif self.parameter == "total":
            return np.asarray(np.sum(rep_weights * y[:, None], axis=0))
        elif self.parameter == "ratio" and x is not None:
            return np.asarray(
                np.sum(rep_weights * y[:, None], axis=0) / np.sum(rep_weights * x[:, None], axis=0)
            )
        else:
            raise AssertionError("Parameter not valid!")

    def _bias(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        rep_weights: np.ndarray,
        x: Optional[np.ndarray],
    ) -> Number:

        estimate = self._get_point(y, samp_weight, x)
        if isinstance(estimate, (int, float)):
            rep_estimates = self._rep_point(y, rep_weights, x)
            return float(np.sum(np.mean(rep_estimates) - estimate))
        else:
            raise AssertionError

    def _rep_coefs(self, rep_coefs: Optional[Union[np.ndarray, Number]] = None) -> None:

        if rep_coefs is not None:
            if isinstance(rep_coefs, np.ndarray):
                self.rep_coefs = rep_coefs
            elif isinstance(rep_coefs, (int, float)):
                self.rep_coefs = rep_coefs
        elif self.number_reps is not None and self.method == "bootstrap":
            self.rep_coefs = (1 / self.number_reps) * np.ones(self.number_reps)
        elif self.number_reps is not None and self.method == "brr":
            self.rep_coefs = (1 / (self.number_reps * pow(1 - self.fay_coef, 2))) * np.ones(
                self.number_reps
            )
        elif self.number_reps is not None and self.method == "jackknife":
            self.rep_coefs = ((self.number_reps - 1) / self.number_reps) * np.ones(
                self.number_reps
            )

    def _variance(
        self,
        y: np.ndarray,
        rep_weights: np.ndarray,
        rep_coefs: np.ndarray,
        x: Optional[np.ndarray],
        estimate: float,
        conservative: bool = False,
    ) -> Number:

        variance = 0.0
        rep_estimates = self._rep_point(y, rep_weights, x)
        if self.method == "jackknife":  # page 155 (4.2.3 and 4.2.5) - Wolter(2003)
            jk_factor = np.array(1 / (1 - rep_coefs))
            pseudo_estimates = jk_factor * estimate - (jk_factor - 1) * rep_estimates
            if conservative:
                variance = float(
                    np.sum(rep_coefs * pow((pseudo_estimates - estimate) / (jk_factor - 1), 2))
                )
            elif not conservative:
                variance = float(
                    np.sum(
                        rep_coefs
                        * pow(
                            (pseudo_estimates - np.mean(pseudo_estimates)) / (jk_factor - 1),
                            2,
                        )
                    )
                )
        else:
            if conservative:
                variance = float(np.sum(rep_coefs * pow(rep_estimates - estimate, 2)))
            elif not conservative:
                variance = float(
                    np.sum(rep_coefs * pow(rep_estimates - np.mean(rep_estimates), 2))
                )
        return variance

    def _get_bias(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        rep_weights: np.ndarray,
        x: Optional[np.ndarray] = None,
        domain: Optional[np.ndarray] = None,
        remove_nan: bool = False,
    ) -> Any:

        if remove_nan:
            if self.parameter == "ratio" and x is not None:
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, samp_weight, x, stratum, domain, _, _ = formats.remove_nans(
                excluded_units, y, samp_weight, x, None, domain, None, None
            )
            rep_weights = rep_weights[~excluded_units, :]

        if self.parameter == "proportion":
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values

        bias: Any
        if domain is None:
            if self.parameter == "proportion" and categories is not None:
                cat_dict = dict()
                for k in range(categories.size):
                    cat_dict_k = dict(
                        {
                            categories[k]: self._bias(
                                y_dummies[:, k], samp_weight, rep_weights, None
                            )
                        }
                    )
                    cat_dict.update(cat_dict_k)
                bias = cat_dict
            else:
                bias = self._bias(y, samp_weight, rep_weights, x)
        else:
            for d in np.unique(domain):
                samp_weight_d = samp_weight * (domain == d)
                rep_weights_d = rep_weights * (domain == d)[:, None]
                if self.parameter == "ratio":
                    x_d = x * (domain == d)
                else:
                    x_d = x
                if self.parameter == "proportion":
                    y_dummies_d = y_dummies * (domain == d)[:, None]
                    cat_dict = dict()
                    for k in range(categories.size):
                        cat_dict_d_k = dict(
                            {
                                categories[k]: self._bias(
                                    y_dummies_d[:, k],
                                    samp_weight_d,
                                    rep_weights_d,
                                    None,
                                )
                            }
                        )
                        cat_dict.update(cat_dict_d_k)
                    bias[d] = cat_dict
                else:
                    y_d = y * (domain == d)
                    bias[d] = self._bias(y_d, samp_weight_d, rep_weights, x_d)

        return bias

    def _get_variance(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        rep_weights: np.ndarray,
        rep_coefs: np.ndarray,
        x: Optional[np.ndarray] = None,
        domain: Optional[np.ndarray] = None,
        conservative: bool = False,
        remove_nan: bool = False,
    ) -> Any:

        if self.parameter == "proportion":
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values

        if domain is None:
            if self.parameter == "proportion":
                estimate_k: dict
                cat_dict = dict()
                for k in range(categories.size):
                    estimate_k = self._get_point(y_dummies[:, k], samp_weight, x)
                    cat_dict_k = dict(
                        {
                            categories[k]: self._variance(
                                y_dummies[:, k],
                                rep_weights,
                                rep_coefs,
                                x,
                                estimate_k[1],
                                conservative,
                            )
                        }
                    )
                    cat_dict.update(cat_dict_k)
                return cat_dict
            else:
                estimate = self._get_point(y, samp_weight, x)
                return self._variance(y, rep_weights, rep_coefs, x, estimate, conservative)
        else:
            variance_else1 = {}
            variance_else2 = {}
            for d in np.unique(domain):
                samp_weight_d = samp_weight * (domain == d)
                rep_weights_d = rep_weights * (domain == d)[:, None]
                if self.parameter == "ratio":
                    x_d = x * (domain == d)
                else:
                    x_d = x
                if self.parameter == "proportion" and categories is not None:
                    y_dummies_d = np.asarray(y_dummies * (domain == d)[:, None])
                    cat_dict = dict()
                    for k in range(categories.size):
                        estimate_d_k = self._get_point(y_dummies_d[:, k], samp_weight_d, x_d)[1]
                        cat_dict_d_k = dict(
                            {
                                categories[k]: self._variance(
                                    y_dummies_d[:, k],
                                    rep_weights_d,
                                    rep_coefs,
                                    x_d,
                                    estimate_d_k,
                                    conservative,
                                )
                            }
                        )
                        cat_dict.update(cat_dict_d_k)
                    variance_else1[d] = cat_dict
                else:
                    y_d = y * (domain == d)
                    estimate_d = self._get_point(y_d, samp_weight_d, x_d)
                    variance_else2[d] = self._variance(
                        y_d,
                        rep_weights_d,
                        rep_coefs,
                        x_d,
                        estimate_d,
                        conservative,
                    )
            if self.parameter == "proportion":
                return variance_else1
            else:
                return variance_else2

    def _get_confint(
        self,
        parameter: str,
        estimate: Any,
        variance: Any,
        quantile: float,
    ) -> Tuple[Any, Any]:

        lower_ci = {}
        upper_ci = {}
        if self.domains is None:
            if parameter == "proportion":
                for level in variance:
                    point_est = estimate[level]
                    std_est = pow(variance[level], 0.5)
                    location_ci = math.log(point_est / (1 - point_est))
                    scale_ci = std_est / (point_est * (1 - point_est))
                    ll = location_ci - quantile * scale_ci
                    lower_ci[level] = math.exp(ll) / (1 + math.exp(ll))
                    uu = location_ci + quantile * scale_ci
                    upper_ci[level] = math.exp(uu) / (1 + math.exp(uu))
                return lower_ci, upper_ci
            else:
                return estimate - quantile * pow(variance, 0.5), estimate + quantile * pow(
                    variance, 0.5
                )

        else:
            lower_ci_else1 = {}
            upper_ci_else1 = {}
            lower_ci_else2 = {}
            upper_ci_else2 = {}
            for key in variance:
                if parameter == "proportion":
                    lower_ci_k = {}
                    upper_ci_k = {}
                    for level in variance[key]:
                        point_est = estimate[key][level]
                        std_est = pow(variance[key][level], 0.5)
                        location_ci = math.log(point_est / (1 - point_est))
                        scale_ci = std_est / (point_est * (1 - point_est))
                        ll = location_ci - quantile * scale_ci
                        lower_ci_k[level] = math.exp(ll) / (1 + math.exp(ll))
                        uu = location_ci + quantile * scale_ci
                        upper_ci_k[level] = math.exp(uu) / (1 + math.exp(uu))
                    lower_ci_else1[key] = lower_ci_k
                    upper_ci_else1[key] = upper_ci_k
                else:
                    lower_ci_else2[key] = estimate[key] - quantile * pow(variance[key], 0.5)
                    upper_ci_else2[key] = estimate[key] + quantile * pow(variance[key], 0.5)

            if self.parameter == "proportion":
                return lower_ci_else1, upper_ci_else1
            else:
                return lower_ci_else2, upper_ci_else2

    def _get_coefvar(
        self,
        parameter: str,
        estimate: Any,  # Any := Union[dict[StringNumber, DictStrNum], DictStrNum, Number]
        variance: Any,
    ) -> Any:  # Any := Union[dict[StringNumber, DictStrNum], DictStrNum, Number]

        if self.domains is None:
            if parameter == "proportion":
                coef_var = {}
                for level in variance:
                    coef_var[level] = pow(variance[level], 0.5) / estimate[level]
            else:
                coef_var = pow(variance, 0.5) / estimate
        else:
            coef_var = {}
            for key in variance:
                if parameter == "proportion":
                    coef_var_k = {}
                    for level in variance[key]:
                        coef_var_k[level] = pow(variance[key][level], 0.5) / estimate[key][level]
                    coef_var[key] = coef_var_k
                else:
                    coef_var[key] = pow(variance[key], 0.5) / estimate[key]
        return coef_var

    def estimate(
        self: TypeRepEst,
        y: Array,
        samp_weight: Array,
        rep_weights: Union[np.ndarray, pd.DataFrame],
        x: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        rep_coefs: Optional[Union[float, np.ndarray]] = None,
        domain: Optional[np.ndarray] = None,
        conservative: bool = False,
        deff: bool = False,  # Todo
        remove_nan: bool = False,
    ) -> TypeRepEst:
        """[summary]

        Args:
            self (TypeRepEst): [description]
            y (Array): [description]
            samp_weight (Array): [description]
            rep_weights (Union[np.ndarray, pd.DataFrame]): [description]
            x (Union[np.ndarray, pd.DataFrame, None], optional): [description]. Defaults to None.
            rep_coefs (Union[float, np.ndarray, None], optional): [description]. Defaults to None.
            domain (Optional[np.ndarray], optional): [description]. Defaults to None.
            conservative (bool, optional): [description]. Defaults to False.
            deff (bool, optional): [description]. Defaults to False.

        Raises:
            AssertionError: [description]

        Returns:
            TypeRepEst: [description]
        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("x must be provided for ratio estimation.")

        y = formats.numpy_array(y)
        samp_weight = formats.numpy_array(samp_weight)
        rep_weights = formats.numpy_array(rep_weights)
        x = formats.numpy_array(x) if x is not None else None

        if remove_nan:
            if self.parameter == "ratio" and x is not None:
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, samp_weight, x, domain = formats.remove_nans(
                excluded_units,
                y,
                samp_weight,
                x,
                domain,
            )
            rep_weights = rep_weights[~excluded_units, :]

        self.conservative = conservative

        if self.number_reps is None:
            self.number_reps = rep_weights.shape[1]

        self._rep_coefs(rep_coefs)

        if domain is not None:
            self.domains = np.unique(domain)
        else:
            self.domains = None

        self.point_est = self._get_point(y, samp_weight, x, domain)
        self.variance = self._get_variance(
            y,
            samp_weight,
            rep_weights,
            np.array(self.rep_coefs),
            x,
            domain,
            conservative,
            remove_nan,
        )

        if self.method == "brr" and self.degree_of_freedom is None:
            self.degree_of_freedom = int(self.number_reps / 2)
        elif self.degree_of_freedom is None:
            self.degree_of_freedom = int(self.number_reps) - 1

        t_quantile = student.ppf(1 - self.alpha / 2, df=self.degree_of_freedom)

        self.lower_ci, self.upper_ci = self._get_confint(
            self.parameter, self.point_est, self.variance, t_quantile
        )
        self.coef_var = self._get_coefvar(self.parameter, self.point_est, self.variance)

        if self.domains is None:
            if self.parameter == "proportion":
                for level in self.variance:
                    self.stderror[level] = pow(self.variance[level], 0.5)
            elif isinstance(self.variance, (int, float)):
                self.stderror = pow(self.variance, 0.5)
        else:
            for key in self.variance:
                if self.parameter == "proportion":
                    stderror = {}
                    for level in self.variance[key]:
                        stderror[level] = pow(self.variance[key][level], 0.5)
                    self.stderror[key] = stderror
                else:
                    self.stderror[key] = pow(self.variance[key], 0.5)

        return self
