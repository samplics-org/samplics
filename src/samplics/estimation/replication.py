"""Estimation of linear parameters
"""
# Author: Mamadou S Diallo <msdiallo@QuantifyAfrica.org>
#
# License: MIT

from typing import Any, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd

import math

from scipy.stats import norm as normal
from scipy.stats import t as student

from samplics.utils import checks, formats

from samplics.estimation.expansion import _SurveyEstimator

EstimateType = Any  # Dict[np.ndarray, Union[Dict[np.ndarray, float], float]]


class ReplicateEstimator(_SurveyEstimator):
    """
    Functions to compute replicate-based uncertainty
    """

    def __init__(
        self,
        method: str,
        parameter: str,
        rep_weight_cls: Optional[Any] = None,
        fay_coef=None,
        alpha: float = 0.05,
        random_seed=None,
    ) -> None:
        """Initializes the instance """

        if method.lower() not in ("bootstrap", "brr", "jackknife"):
            raise ValueError("method must be 'bootstrap', 'brr', or 'jackknife'")

        super().__init__(parameter, alpha, random_seed)
        self.method = method.lower()
        self.conservative = False
        self.degree_of_freedom = None
        self.rep_coefs = None
        if method == "brr" and fay_coef is not None:
            self.fay_coef = fay_coef
        elif method == "brr":
            self.fay_coef = 0
        if rep_weight_cls is not None:
            # self.method = rep_weight_cls.rep_method
            self.number_reps = rep_weight_cls.number_reps
            self.rep_coefs = rep_weight_cls.rep_coefs
            self.degree_of_freedom = rep_weight_cls.degree_of_freedom
            self.fay_coef = rep_weight_cls.fay_coef if self.method == "brr" else None

    def _rep_point(self, y: np.ndarray, rep_weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.parameter in ("proportion", "mean"):
            return np.sum(rep_weights * y[:, None], axis=0) / np.sum(rep_weights, axis=0)
        elif self.parameter == "total":
            return np.sum(rep_weights * y[:, None], axis=0)
        elif self.parameter == "ratio":
            return np.sum(rep_weights * y[:, None], axis=0) / np.sum(
                rep_weights * x[:, None], axis=0
            )

    def _bias(
        self, y: np.ndarray, samp_weight: np.ndarray, rep_weights: np.ndarray, x: np.ndarray
    ) -> float:

        estimate = self.get_point(y, samp_weight, x).get("__none__")
        rep_estimates = self._rep_point(y, rep_weights, x)

        return float(np.sum(np.mean(rep_estimates) - estimate))

    def _rep_coefs(self, rep_coefs: np.ndarray = None) -> None:

        if rep_coefs is not None:
            if isinstance(rep_coefs, np.ndarray):
                self.rep_coefs = rep_coefs
            elif isinstance(rep_coefs, (int, float)):
                self.rep_coefs = [rep_coefs]
        elif self.rep_coefs is None and self.method == "bootstrap":
            self.rep_coefs = (1 / self.number_reps) * np.ones(self.number_reps)
        elif self.rep_coefs is None and self.method == "brr":
            self.rep_coefs = (1 / (self.number_reps * pow(1 - self.fay_coef, 2))) * np.ones(
                self.number_reps
            )
        elif self.rep_coefs is None and self.method == "jackknife":
            self.rep_coefs = ((self.number_reps - 1) / self.number_reps) * np.ones(
                self.number_reps
            )

    def _variance(
        self,
        y: np.ndarray,
        rep_weights: np.ndarray,
        rep_coefs: np.ndarray,
        x: np.ndarray,
        estimate: float,
        conservative: bool = False,
    ) -> float:

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
                        * pow((pseudo_estimates - np.mean(pseudo_estimates)) / (jk_factor - 1), 2)
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
        domain: np.ndarray = None,
        exclude_nan: bool = False,
    ) -> EstimateType:
        """
        estimate bias using replication methods. 

        Args:
            y (array) : 

            samp_weight (array):

            re_weights (array):

            domain (array):    
        
        Returns:
            A dictionary: .
        """

        if exclude_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, samp_weight, x, stratum, domain, _, _ = self._exclude_nans(
                excluded_units, y, samp_weight, x, None, domain, None, None
            )
            rep_weights = rep_weights[~excluded_units, :]

        if self.parameter == "proportion":
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values

        bias: EstimateType = {}
        if domain is None:
            if self.parameter == "proportion":
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
                bias["__none__"] = cat_dict
            else:
                bias["__none__"] = self._bias(y, samp_weight, rep_weights, x)
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
                                    y_dummies_d[:, k], samp_weight_d, rep_weights, None
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
        x: np.ndarray = None,
        domain: np.ndarray = None,
        conservative: bool = False,
        exclude_nan: bool = False,
    ) -> EstimateType:
        """
        estimate variance using replication methods. 

        Args:
            y (array) : 

            samp_weight (array):

            rep_weights (array):

            rep_coefs (array):

            domain (array):    
        
        Returns:
            A dictionary: .
        """

        if self.parameter == "proportion":
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values

        variance: EstimateType = {}
        if domain is None:
            if self.parameter == "proportion":
                cat_dict = dict()
                for k in range(categories.size):
                    estimate_k = self.get_point(y_dummies[:, k], samp_weight, x)["__none__"][1]
                    cat_dict_k = dict(
                        {
                            categories[k]: self._variance(
                                y_dummies[:, k],
                                rep_weights,
                                rep_coefs,
                                x,
                                estimate_k,
                                conservative,
                            )
                        }
                    )
                    cat_dict.update(cat_dict_k)
                variance["__none__"] = cat_dict
            else:
                estimate = self.get_point(y, samp_weight, x).get("__none__")
                variance["__none__"] = self._variance(
                    y, rep_weights, rep_coefs, x, estimate, conservative
                )
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
                        estimate_d_k = self.get_point(y_dummies_d[:, k], samp_weight_d, x_d).get(
                            "__none__"
                        )[1]
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
                    variance[d] = cat_dict
                else:
                    y_d = y * (domain == d)
                    estimate_d = self.get_point(y_d, samp_weight_d, x_d).get("__none__")
                    variance[d] = self._variance(
                        y_d, rep_weights_d, rep_coefs, x_d, estimate_d, conservative
                    )

        return variance

    @staticmethod
    def _get_confint(
        parameter: str, estimate: EstimateType, variance: EstimateType, quantile: float
    ) -> Tuple[EstimateType, EstimateType]:
        """
        estimate variance using replication methods. 

        Args:
            paremeter: 

        
        Returns:
            A dictionary: .
        """

        lower_ci: EstimateType = {}
        upper_ci: EstimateType = {}
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
                lower_ci[key] = lower_ci_k
                upper_ci[key] = upper_ci_k
            else:
                lower_ci[key] = estimate[key] - quantile * pow(variance[key], 0.5)
                upper_ci[key] = estimate[key] + quantile * pow(variance[key], 0.5)

        return lower_ci, upper_ci

    @staticmethod
    def _get_coefvar(
        parameter: str, estimate: EstimateType, variance: EstimateType
    ) -> EstimateType:
        """Computes the coefficient of variation

        Args:

        parameter:

        Returns:
        A float or dictionnary: 

        """

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
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        rep_weights: np.ndarray,
        x: Optional[np.ndarray] = None,
        rep_coefs: Optional[Union[float, np.ndarray]] = None,
        domain: Optional[np.ndarray] = None,
        conservative: bool = False,
        deff: bool = False,  # Todo
        exclude_nan: bool = False,
    ) -> EstimateType:
        """Computes the parameter point estimates

        Args:

            y:

            samp_weights:

            rep_weights: 

            domain:

        Returns:
        A ..

        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("x must be provided for ratio estimation.")

        if not isinstance(rep_weights, np.ndarray):
            rep_weights = formats.numpy_array(rep_weights)

        if exclude_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, samp_weight, x, _, domain, _, _ = self._exclude_nans(
                excluded_units, y, samp_weight, x, None, domain, None, None
            )
            rep_weights = rep_weights[~excluded_units, :]

        self.conservative = conservative

        if self.number_reps is None:
            self.number_reps = rep_weights.shape[1]

        self._rep_coefs(rep_coefs)

        if domain is not None:
            self.domains = np.unique(domain)

        self.point_est = self.get_point(y, samp_weight, x, domain)
        self.variance = self._get_variance(
            y,
            samp_weight,
            rep_weights,
            np.array(self.rep_coefs),
            x,
            domain,
            conservative,
            exclude_nan,
        )

        if self.method == "brr" and self.degree_of_freedom is None:
            self.degree_of_freedom = self.number_reps / 2
        elif self.degree_of_freedom is None:
            self.degree_of_freedom = self.number_reps - 1

        t_quantile = student.ppf(1 - self.alpha / 2, df=self.degree_of_freedom)

        self.lower_ci, self.upper_ci = self._get_confint(
            self.parameter, self.point_est, self.variance, t_quantile
        )
        self.coef_var = self._get_coefvar(self.parameter, self.point_est, self.variance)

        for key in self.variance:
            if self.parameter == "proportion":
                stderror = {}
                for level in self.variance[key]:
                    stderror[level] = pow(self.variance[key][level], 0.5)
                self.stderror[key] = stderror
            else:
                self.stderror[key] = pow(self.variance[key], 0.5)

        return self
