"""Expansion module 

The module implements the taylor-based variance estimation methods. There are several good books
that provide the the taylor-based approximations of the sample variance. Some of these reference 
books are Cochran, W.G. (1977) [#c1977]_, Kish, L. (1965) [#k1965]_, Lohr, S.L. (2010) [#l2010]_, 
and Wolter, K.M. (2007) [#w2007]_. 

.. [#c1977] Cochran, W.G. (1977), *Sampling Techniques, 3rd edn.*, Jonh Wiley & Sons, Inc.
.. [#k1965] Kish, L. (1965), *Survey Sampling*, Jonh Wiley & Sons, Inc.
.. [#l2010] Lohr, S.L. (2010), *Sampling: Design and Analysis, 2nd edn.*, Cengage Learning, Inc.
.. [#w2007] Wolter, K.M. (2007), *Introduction to Variance Estimation, 2nd edn.*, 
   Springer-Verlag New York, Inc
"""

import math
from typing import TypeVar, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from samplics.utils import formats
from samplics.utils.types import Array, StringNumber
from scipy.stats import t as student

TypeTaylorEst = TypeVar("TypeTaylorEst", bound="TaylorEstimator")


class _SurveyEstimator:
    """General approach for sample estimation of linear parameters."""

    def __init__(self, parameter: str, alpha: float = 0.05, random_seed: int = None) -> None:
        """Initializes the instance """

        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(self.random_seed)
        else:
            self.random_seed = None

        if parameter.lower() in ("proportion", "mean", "total", "ratio"):
            self.parameter = parameter.lower()
        else:
            raise AssertionError("parameter must be 'proportion', 'mean', 'total' or 'ratio'")

        self.point_est: Dict[StringNumber, Any] = {}
        self.variance: Dict[StringNumber, Any] = {}
        self.stderror: Dict[StringNumber, Any] = {}
        self.coef_var: Dict[StringNumber, Any] = {}
        self.deff: Dict[StringNumber, Any] = {}
        self.lower_ci: Dict[StringNumber, Any] = {}
        self.upper_ci: Dict[StringNumber, Any] = {}
        self.strata: List[StringNumber] = ["__none__"]
        self.domains: List[StringNumber] = ["__none__"]
        self.method: str = "taylor"
        self.number_strata: Optional[int] = None
        self.number_psus: Optional[int] = None
        self.degree_of_freedom: Optional[int] = None
        self.alpha: float = alpha

    def __str__(self) -> Any:
        print(f"SAMPLICS - Estimation of {self.parameter.title()}\n")
        print(f"Number of strata: {self.number_strata}")
        print(f"Number of psus: {self.number_psus}")
        print(f"Degree of freedom: {self.degree_of_freedom}\n")
        parameter = self.parameter.upper()
        estimation = pd.DataFrame()
        estimation["DOMAINS"] = self.domains
        estimation[parameter] = self.point_est.values()
        estimation["SE"] = self.stderror.values()
        estimation["LCI"] = self.lower_ci.values()
        estimation["UCI"] = self.upper_ci.values()
        estimation["CV"] = self.coef_var.values()

        return "%s" % estimation

    def __repr__(self) -> Any:
        return self.__str__()

    def _remove_nans(
        self,
        excluded_units: Array,
        y: Array,
        samp_weight: Array,
        x: Array = None,
        stratum: Array = None,
        domain: Array = None,
        psu: Array = None,
        ssu: Array = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        y = formats.numpy_array(y)
        samp_weight = formats.numpy_array(samp_weight)
        if x is not None:
            x = formats.numpy_array(x)
            x = x[~excluded_units]
        if stratum is not None:
            stratum = formats.numpy_array(stratum)
            stratum = stratum[~excluded_units]
        if domain is not None:
            domain = formats.numpy_array(domain)
            domain = domain[~excluded_units]
        if psu is not None:
            psu = formats.numpy_array(psu)
            psu = psu[~excluded_units]
        if ssu is not None:
            ssu = formats.numpy_array(ssu)
            ssu = ssu[~excluded_units]

        return (
            y[~excluded_units],
            samp_weight[~excluded_units],
            x,
            stratum,
            domain,
            psu,
            ssu,
        )

    def _degree_of_freedom(
        self,
        samp_weight: np.ndarray,
        stratum: np.ndarray = None,
        psu: np.ndarray = None,
    ) -> None:

        stratum = formats.numpy_array(stratum)
        psu = formats.numpy_array(psu)

        if stratum.size <= 1:
            self.number_psus = np.unique(psu).size if psu.size > 1 else samp_weight.size
            self.number_strata = 1
        elif psu.size > 1:
            self.number_psus = np.unique([stratum, psu], axis=1).shape[1]
            self.number_strata = np.unique(stratum).size
        else:
            samp_weight = formats.numpy_array(samp_weight)
            self.number_psus = samp_weight.size
            self.number_strata = np.unique(stratum).size

        self.degree_of_freedom = self.number_psus - self.number_strata

    def _get_point_d(
        self, y: np.ndarray, samp_weight: np.ndarray, x: np.ndarray = None
    ) -> np.float64:

        if self.parameter in ("proportion", "mean"):
            return np.sum(samp_weight * y) / np.sum(samp_weight)
        elif self.parameter == "total":
            return np.sum(samp_weight * y)
        elif self.parameter == "ratio":
            return np.sum(samp_weight * y) / np.sum(samp_weight * x)

    def _get_point(
        self,
        y: Array,
        samp_weight: Array,
        x: Optional[Array] = None,
        domain: Optional[Array] = None,
        remove_nan: bool = False,
    ) -> Dict[StringNumber, Any]:
        """Computes the parameter point estimates

        Args:

        y:

        samp_weight:

        domain:

        Returns:
        A float or dictionary: The estimated parameter

        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("Parameter x must be provided for ratio estimation.")

        if remove_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, samp_weight, x, _, domain, _, _ = self._remove_nans(
                excluded_units, y, samp_weight, x, domain=domain
            )

        if self.parameter == "proportion":
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values
        else:
            y_dummies = None
            categories = None
            y_dummies = None

        estimate: Dict[StringNumber, Any] = {}
        if domain is None:
            if self.parameter == "proportion":
                cat_dict = dict()
                for k in range(categories.size):
                    y_k = y_dummies[:, k]
                    cat_dict_k = dict({categories[k]: self._get_point_d(y_k, samp_weight)})
                    cat_dict.update(cat_dict_k)
                estimate["__none__"] = cat_dict
            else:
                estimate["__none__"] = self._get_point_d(y, samp_weight, x)
        else:
            domain_ids = np.unique(domain)
            for d in domain_ids:
                weight_d = samp_weight[domain == d]
                if self.parameter == "proportion":
                    cat_dict = dict()
                    for k in range(categories.size):
                        y_d_k = y_dummies[domain == d, k]
                        cat_dict_d_k = dict({categories[k]: self._get_point_d(y_d_k, weight_d)})
                        cat_dict.update(cat_dict_d_k)
                    estimate[d] = cat_dict
                else:
                    y_d = y[domain == d]
                    x_d = x[domain == d] if self.parameter == "ratio" else None
                    estimate[d] = self._get_point_d(y_d, weight_d, x_d)

        return estimate


class TaylorEstimator(_SurveyEstimator):
    """*TaylorEstimate* implements taylor-based variance approximations.

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
        | number_psus (int): number of primary sampling units (psus)

    Methods
        | estimate(): produces the point estimate of the parameter of interest with the associated
        |   measures of precision.
    """

    def __init__(
        self,
        parameter: str,
        alpha: float = 0.05,
        random_seed: int = None,
        ciprop_method: str = "logit",
    ) -> None:
        """Initializes the instance """
        _SurveyEstimator.__init__(self, parameter)
        if self.parameter == "proportion":
            self.ciprop_method = ciprop_method

    def _score_variable(
        self, y: np.ndarray, samp_weight: np.ndarray, x: np.ndarray = None
    ) -> np.ndarray:
        """Provides the scores used to calculate the variance"""

        y_weighted = y * samp_weight
        if self.parameter in ("proportion", "mean"):
            scale_weights = np.sum(samp_weight)
            location_weights = np.sum(y_weighted) / scale_weights
            return samp_weight * (y - location_weights) / scale_weights
        elif self.parameter == "ratio":
            weighted_sum_x = np.sum(x * samp_weight)
            weighted_ratio = np.sum(y_weighted) / weighted_sum_x
            return samp_weight * (y - x * weighted_ratio) / weighted_sum_x
        elif self.parameter == "total":
            return y_weighted
        else:
            raise ValueError("parameter not valid!")

    @staticmethod
    def _variance_stratum_between(
        y_score_s: np.ndarray, samp_weight_s: np.ndarray, number_psus_in_s: int, psu_s: np.ndarray
    ) -> np.float64:
        """Computes the variance for one stratum """

        variance = 0.0
        if number_psus_in_s > 1:
            scores_s_mean = y_score_s.sum() / number_psus_in_s
            psus = np.unique(psu_s)
            scores_psus_sums = np.zeros(number_psus_in_s)
            for k, psu in enumerate(np.unique(psus)):
                scores_psus_sums[k] = y_score_s[psu_s == psu].sum()
            variance = ((scores_psus_sums - scores_s_mean) ** 2).sum()
            variance = (number_psus_in_s / (number_psus_in_s - 1)) * variance
        elif number_psus_in_s in (0, 1):
            number_obs = y_score_s.size
            y_score_s_mean = y_score_s.sum() / number_obs
            variance = (number_obs / (number_obs - 1)) * ((y_score_s - y_score_s_mean) ** 2).sum()
        else:
            raise ValueError("Number of psus cannot be negative.")

        return variance

    @staticmethod
    def _variance_stratum_within(
        y_score_s: np.ndarray,
        number_psus_in_s: int,
        psu_s: np.ndarray,
        ssu_s: np.ndarray,
    ) -> np.float64:

        variance = 0.0

        if ssu_s is not None:
            psus = np.unique(psu_s)
            for psu in np.unique(psus):
                scores_psu_mean = y_score_s[psus == psu].mean()
                ssus = np.unique(ssu_s[psu_s == psu])
                number_ssus_in_psu = np.size(ssus)
                scores_ssus_sums = np.zeros(number_ssus_in_psu)
                if number_ssus_in_psu > 1:
                    for k, ssu in enumerate(np.unique(ssus)):
                        scores_ssus_sums[k] = y_score_s[ssu_s == ssu].sum()
                    variance += (number_ssus_in_psu / (number_ssus_in_psu - 1)) * (
                        (scores_ssus_sums - scores_psu_mean) ** 2
                    ).sum()

        return variance

    def _taylor_variance(
        self,
        y_score: np.ndarray,
        samp_weight: np.ndarray,
        stratum: np.ndarray,
        psu: np.ndarray,
        ssu: Optional[np.ndarray] = None,
    ) -> np.float64:
        """Computes the variance across stratum """

        if stratum is None:
            number_psus = np.unique(psu).size
            var_est = self._variance_stratum_between(
                y_score, samp_weight, number_psus, psu
            ) + self._variance_stratum_within(y_score, number_psus, psu, ssu)

        else:
            var_est = 0.0
            for s in np.unique(stratum):
                y_score_s = y_score[stratum == s]
                samp_weight_s = samp_weight[stratum == s]
                psu_s = psu[stratum == s] if psu is not None else np.array([])
                number_psus_in_s = np.size(np.unique(psu_s))
                ssu_s = ssu[stratum == s] if ssu is not None else None
                var_est += self._variance_stratum_between(
                    y_score_s, samp_weight_s, number_psus_in_s, psu_s
                ) + self._variance_stratum_within(y_score_s, number_psus_in_s, psu_s, ssu_s)

        return var_est

    def _get_variance(
        self,
        y: Array,
        samp_weight: Array,
        x: Optional[Array] = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        domain: Optional[Array] = None,
        remove_nan: bool = False,
    ) -> Dict[StringNumber, Any]:

        if self.parameter == "ratio" and x is None:
            raise AssertionError("Parameter x must be provided for ratio estimation.")

        if remove_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, samp_weight, x, stratum, domain, psu, ssu = self._remove_nans(
                excluded_units, y, samp_weight, x, stratum, domain, psu, ssu
            )

        if self.parameter == "proportion":
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values
        else:
            y_dummies = None
            categories = None
            y_dummies = None

        variance: Dict[StringNumber, Any] = {}
        if domain is None:
            if self.parameter == "proportion":
                cat_dict = dict()
                for k in range(categories.size):
                    y_score_k = self._score_variable(y_dummies[:, k], samp_weight)
                    cat_dict_k = dict(
                        {
                            categories[k]: self._taylor_variance(
                                y_score_k, samp_weight, stratum, psu, ssu
                            )
                        }
                    )
                    cat_dict.update(cat_dict_k)
                variance["__none__"] = cat_dict
            else:
                y_score = self._score_variable(y, samp_weight, x)
                variance[self.domains[0]] = self._taylor_variance(
                    y_score, samp_weight, stratum, psu, ssu
                )

        else:
            for d in np.unique(domain):
                weight_d = samp_weight * (domain == d)
                if self.parameter == "ratio":
                    x_d = x * (domain == d)
                else:
                    x_d = x
                if self.parameter == "proportion":
                    y_dummies_d = y_dummies * (domain == d)[:, None]
                    cat_dict = dict()
                    for k in range(categories.size):
                        y_score_d_k = self._score_variable(y_dummies_d[:, k], weight_d)
                        cat_dict_d_k = dict(
                            {
                                categories[k]: self._taylor_variance(
                                    y_score_d_k, weight_d, stratum, psu, ssu
                                )
                            }
                        )
                        cat_dict.update(cat_dict_d_k)
                    variance[d] = cat_dict
                else:
                    y_d = y * (domain == d)
                    y_score_d = self._score_variable(y_d, weight_d, x_d)
                    variance[d] = self._taylor_variance(y_score_d, weight_d, stratum, psu, ssu)

        return variance

    def estimate(
        self: TypeTaylorEst,
        y: Array,
        samp_weight: Array,
        x: Optional[Array] = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        domain: Optional[Array] = None,
        deff: bool = False,
        coef_variation: bool = False,
        remove_nan: bool = False,
    ) -> None:
        """[summary]

        Args:
            self (TypeTaylorEst): [description]
            y (Array): [description]
            samp_weight (Array): [description]
            x (Optional[Array], optional): [description]. Defaults to None.
            stratum (Optional[Array], optional): [description]. Defaults to None.
            psu (Optional[Array], optional): [description]. Defaults to None.
            ssu (Optional[Array], optional): [description]. Defaults to None.
            domain (Optional[Array], optional): [description]. Defaults to None.
            deff (bool, optional): [description]. Defaults to False.
            coef_variation (bool, optional): [description]. Defaults to False.
            remove_nan (bool, optional): [description]. Defaults to False.

        Raises:
            AssertionError: [description]

        Returns:
            TypeTaylorEst: [description]
        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("x must be provided for ratio estimation.")

        if remove_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, samp_weight, x, stratum, domain, psu, ssu = self._remove_nans(
                excluded_units, y, samp_weight, x, stratum, domain, psu, ssu
            )

        if stratum is not None:
            self.strata = np.unique(stratum).tolist()
        if domain is not None:
            self.domains = np.unique(domain).tolist()

        self.point_est = self._get_point(y, samp_weight, x, domain)
        self.variance = self._get_variance(y, samp_weight, x, stratum, psu, ssu, domain)

        self._degree_of_freedom(samp_weight, stratum, psu)
        t_quantile = student.ppf(1 - self.alpha / 2, df=self.degree_of_freedom)

        for key in self.variance:
            if self.parameter == "proportion":
                stderror = {}
                lower_ci = {}
                upper_ci = {}
                coef_var = {}
                for level in self.variance[key]:
                    point_est = self.point_est[key][level]
                    stderror[level] = pow(self.variance[key][level], 0.5)
                    if point_est == 0:
                        lower_ci[level] = 0
                        upper_ci[level] = 0
                        coef_var[level] = 0
                    elif point_est == 1:
                        lower_ci[level] = 1
                        upper_ci[level] = 1
                        coef_var[level] = 0
                    else:
                        location_ci = math.log(point_est / (1 - point_est))
                        scale_ci = stderror[level] / (point_est * (1 - point_est))
                        ll = location_ci - t_quantile * scale_ci
                        uu = location_ci + t_quantile * scale_ci
                        lower_ci[level] = math.exp(ll) / (1 + math.exp(ll))
                        upper_ci[level] = math.exp(uu) / (1 + math.exp(uu))
                        coef_var[level] = stderror[level] / point_est

                self.stderror[key] = stderror
                self.coef_var[key] = coef_var
                self.lower_ci[key] = lower_ci
                self.upper_ci[key] = upper_ci
            else:
                self.stderror[key] = pow(self.variance[key], 0.5)
                self.lower_ci[key] = self.point_est[key] - t_quantile * self.stderror[key]
                self.upper_ci[key] = self.point_est[key] + t_quantile * self.stderror[key]
                self.coef_var[key] = pow(self.variance[key], 0.5) / self.point_est[key]
