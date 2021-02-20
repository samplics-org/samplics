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
from typing import TypeVar, Generic, Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from samplics.utils.formats import fpc_as_dict, numpy_array, remove_nans, sample_size_dict
from samplics.utils.types import Array, Number, Series, StringNumber
from scipy.stats import t as student


class _SurveyEstimator:
    """General approach for sample estimation of linear parameters."""

    def __init__(self, parameter: str, alpha: float = 0.05, random_seed: Optional[int] = None):
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
        self.alpha = alpha

        self.point_est: Any = {}
        self.variance: Any = {}
        self.covariance: Any = {}
        self.stderror: Any = {}
        self.coef_var: Any = {}
        self.deff: Any = {}
        self.lower_ci: Any = {}
        self.upper_ci: Any = {}
        self.fpc: Any = {}
        self.strata: Any = []
        self.domains: Any = None
        self.method: Any = "taylor"
        self.number_strata: Any = None
        self.number_psus: Any = None
        self.degree_of_freedom: Any = None

    def __str__(self) -> Any:
        print(f"SAMPLICS - Estimation of {self.parameter.title()}\n")
        print(f"Number of strata: {self.number_strata}")
        print(f"Number of psus: {self.number_psus}")
        print(f"Degree of freedom: {self.degree_of_freedom}\n")
        parameter = self.parameter.upper()
        estimation = pd.DataFrame()
        if (
            isinstance(self.point_est, dict)
            and isinstance(self.stderror, dict)
            and isinstance(self.lower_ci, dict)
            and isinstance(self.upper_ci, dict)
            and isinstance(self.coef_var, dict)
        ):
            estimation["DOMAINS"] = self.domains
            estimation[parameter] = self.point_est.values()
            estimation["SE"] = self.stderror.values()
            estimation["LCI"] = self.lower_ci.values()
            estimation["UCI"] = self.upper_ci.values()
            estimation["CV"] = self.coef_var.values()
        else:
            estimation[parameter] = [self.point_est]
            estimation["SE"] = [self.stderror]
            estimation["LCI"] = [self.lower_ci]
            estimation["UCI"] = [self.upper_ci]
            estimation["CV"] = [self.coef_var]

        return "%s" % estimation.to_string(index=False)

    def __repr__(self) -> Any:
        return self.__str__()

    def _degree_of_freedom(
        self,
        samp_weight: np.ndarray,
        stratum: np.ndarray = None,
        psu: np.ndarray = None,
    ) -> None:

        stratum = numpy_array(stratum)
        psu = numpy_array(psu)

        if stratum.size <= 1:
            self.number_psus = np.unique(psu).size if psu.size > 1 else samp_weight.size
            self.number_strata = 1
        elif psu.size > 1:
            self.number_psus = np.unique([stratum, psu], axis=1).shape[1]
            self.number_strata = np.unique(stratum).size
        else:
            samp_weight = numpy_array(samp_weight)
            self.number_psus = samp_weight.size
            self.number_strata = np.unique(stratum).size

        self.degree_of_freedom = self.number_psus - self.number_strata

    def _get_point_d(
        self, y: np.ndarray, samp_weight: np.ndarray, x: Optional[np.ndarray] = None
    ) -> float:

        if self.parameter in ("proportion", "mean"):
            return float(np.sum(samp_weight * y) / np.sum(samp_weight))
        elif self.parameter == "total":
            return float(np.sum(samp_weight * y))
        elif self.parameter == "ratio":
            return float(np.sum(samp_weight * y) / np.sum(samp_weight * x))
        else:
            raise ValueError("Parameter not valid!")

    def _get_point(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        x: Optional[np.ndarray] = None,
        domain: Optional[np.newaxis] = None,
        as_factor: bool = False,
        remove_nan: bool = False,
    ) -> Union[Dict[StringNumber, Dict[StringNumber, float]], Dict[StringNumber, float], float]:
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
            y, samp_weight, x, domain = remove_nans(excluded_units, y, samp_weight, x, domain)

        if self.parameter == "proportion" or as_factor:
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values
        else:
            categories = None
            y_dummies = None

        if domain is None:
            if self.parameter == "proportion" or as_factor:
                cat_dict: Dict[StringNumber, float] = {}
                for k in range(categories.size):
                    y_k = y_dummies[:, k]
                    cat_dict_k = dict({categories[k]: self._get_point_d(y_k, samp_weight)})
                    cat_dict.update(cat_dict_k)
                return cat_dict
            else:
                return self._get_point_d(y, samp_weight, x)
        else:
            domain_ids = np.unique(domain)
            if self.parameter == "proportion" or as_factor:
                estimate1: Dict[StringNumber, Dict[StringNumber, float]] = {}
                for d in domain_ids:
                    weight_d = samp_weight[domain == d]
                    cat_dict_d: Dict[StringNumber, float] = {}
                    for k in range(categories.size):
                        y_d_k = y_dummies[domain == d, k]
                        cat_dict_d_k = dict({categories[k]: self._get_point_d(y_d_k, weight_d)})
                        cat_dict_d.update(cat_dict_d_k)
                    estimate1[d] = cat_dict_d
                return estimate1
            else:
                estimate2: Dict[StringNumber, float] = {}
                for d in domain_ids:
                    weight_d = samp_weight[domain == d]
                    y_d = y[domain == d]
                    x_d = x[domain == d] if self.parameter == "ratio" else None
                    estimate2[d] = self._get_point_d(y_d, weight_d, x_d)
                return estimate2


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
        random_seed: Optional[int] = None,
        ciprop_method: Optional[str] = "logit",
    ) -> None:
        """Initializes the instance """
        _SurveyEstimator.__init__(self, parameter)
        if self.parameter == "proportion":
            self.ciprop_method = ciprop_method
        else:
            self.ciprop_method = None

    def _score_variable(
        self, y: np.ndarray, samp_weight: np.ndarray, x: np.ndarray = None
    ) -> np.ndarray:
        """Provides the scores used to calculate the variance"""

        y = np.asarray(y)
        samp_weight = np.asarray(samp_weight)
        x = np.asarray(x)

        ncols = 1 if len(y.shape) == 1 else y.shape[1]
        y = y.reshape(y.shape[0], ncols)
        y_weighted = y * samp_weight[:, None]  # .reshape(samp_weight.shape[0], 1)
        if self.parameter in ("proportion", "mean"):
            scale_weights = np.sum(samp_weight)
            location_weights = np.sum(y_weighted, axis=0) / scale_weights
            return (y - location_weights) * samp_weight[:, None] / scale_weights
        elif self.parameter == "ratio":
            weighted_sum_x = np.sum(x * samp_weight)
            weighted_ratio = np.sum(y_weighted, axis=0) / weighted_sum_x
            return samp_weight[:, None] * (y - x[:, None] * weighted_ratio) / weighted_sum_x
        elif self.parameter == "total":
            return y_weighted
        else:
            raise ValueError("parameter not valid!")

    @staticmethod
    def _variance_stratum_between(
        y_score_s: np.ndarray, samp_weight_s: np.ndarray, number_psus_in_s: int, psu_s: np.ndarray
    ) -> np.ndarray:
        """Computes the variance for one stratum """

        covariance = np.asarray([])
        if number_psus_in_s > 1:
            scores_s_mean = y_score_s.sum(axis=0) / number_psus_in_s  # new
            psus = np.unique(psu_s)
            scores_psus_sums = np.zeros((number_psus_in_s, scores_s_mean.shape[0]))
            for k, psu in enumerate(np.unique(psus)):
                scores_psus_sums[k, :] = y_score_s[psu_s == psu].sum(axis=0)
            covariance = np.transpose(scores_psus_sums - scores_s_mean) @ (
                scores_psus_sums - scores_s_mean
            )
            covariance = (number_psus_in_s / (number_psus_in_s - 1)) * covariance
        elif number_psus_in_s in (0, 1):
            number_obs = y_score_s.shape[0]
            y_score_s_mean = y_score_s.sum(axis=0) / number_obs
            covariance = (
                (number_obs / (number_obs - 1))
                * np.transpose(y_score_s - y_score_s_mean)
                @ (y_score_s - y_score_s_mean)
            )
        else:
            raise ValueError("Number of psus cannot be negative.")

        return covariance

    @staticmethod
    def _variance_stratum_within(
        y_score_s: np.ndarray,
        number_psus_in_s: np.ndarray,
        psu_s: np.ndarray,
        ssu_s: np.ndarray,
    ) -> float:

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
        fpc: Union[Dict[StringNumber, Number], Number] = 1,
    ) -> np.ndarray:
        """Computes the variance across stratum """

        if stratum is None and isinstance(fpc, (int, float)):
            number_psus = np.unique(psu).size
            return fpc * self._variance_stratum_between(y_score, samp_weight, number_psus, psu)
            # + (1 - fpc) * self._variance_stratum_within(
            #     y_score, number_psus, psu, ssu
            # )
        elif isinstance(fpc, dict):
            covariance = np.zeros((y_score.shape[1], y_score.shape[1]))
            for s in np.unique(stratum):
                y_score_s = y_score[stratum == s]
                samp_weight_s = samp_weight[stratum == s]
                psu_s = psu[stratum == s] if psu is not None else None  # np.array([])
                number_psus_in_s = np.size(np.unique(psu_s))
                ssu_s = ssu[stratum == s] if ssu is not None else None
                covariance += fpc[s] * self._variance_stratum_between(
                    y_score_s, samp_weight_s, number_psus_in_s, psu_s
                )
                #  + (1 - fpc[s]) * self._variance_stratum_within(
                #     y_score_s, number_psus_in_s, psu_s, ssu_s
                # )
            return covariance
        else:
            raise TypeError

    def _get_variance(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        x: Optional[np.ndarray] = None,
        stratum: Optional[np.ndarray] = None,
        psu: Optional[np.ndarray] = None,
        ssu: Optional[np.ndarray] = None,
        domain: Optional[np.ndarray] = None,
        fpc: Union[Dict[StringNumber, Number], Number] = 1,
        as_factor: bool = False,
        remove_nan: bool = False,
    ) -> Tuple[
        Union[Dict[StringNumber, Number], np.ndarray],
        Union[Dict[StringNumber, Number], np.ndarray],
    ]:

        if self.parameter == "ratio" and x is None:
            raise AssertionError("Parameter x must be provided for ratio estimation.")

        if remove_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
                y, samp_weight, x, stratum, domain, psu, ssu = remove_nans(
                    excluded_units, y, samp_weight, x, stratum, domain, psu, ssu
                )

        # if isinstance(y, np.ndarray):

        categories = None
        if self.parameter == "proportion" or as_factor:
            y = pd.get_dummies(y).astype(int)
            categories = list(y.columns)
            y = y.values

        if domain is None:
            y_score = self._score_variable(y, samp_weight, x)  # new
            cov_score = self._taylor_variance(y_score, samp_weight, stratum, psu, ssu, fpc)  # new
            if (self.parameter == "proportion" or as_factor) and isinstance(cov_score, np.ndarray):
                variance1: Dict[StringNumber, dict] = {}
                covariance1: Dict[StringNumber, dict] = {}
                variance1 = dict(zip(categories, np.diag(cov_score)))
                for k, level in enumerate(categories):
                    covariance1[level] = dict(zip(categories, cov_score[k, :]))
                return variance1, covariance1
            else:
                return cov_score, cov_score  # Todo: generalize for multiple Y variables
        else:
            domain = np.asarray(domain)
            if self.parameter == "proportion" or as_factor:
                variance2: Dict[StringNumber, dict] = {}
                covariance2: Dict[StringNumber, dict] = {}
                for d in np.unique(domain):
                    domain_d = domain == d
                    weight_d = samp_weight * domain_d
                    if self.parameter == "ratio":
                        x_d = x * domain_d
                    else:
                        x_d = x
                    y_d = y * domain_d if len(y.shape) == 1 else y * domain_d[:, None]
                    y_score_d = self._score_variable(y_d, weight_d, x_d)
                    cov_score_d = self._taylor_variance(
                        y_score_d, weight_d, stratum, psu, ssu, fpc
                    )
                    variance2[d] = dict(zip(categories, np.diag(cov_score_d)))
                    cov_d = {}
                    for k, level in enumerate(categories):
                        cov_d.update({level: dict(zip(categories, cov_score_d[k, :]))})
                    covariance2.update({d: cov_d})
                return variance2, covariance2
            else:
                variance3: Dict[StringNumber, float] = {}
                for d in np.unique(domain):
                    domain_d = domain == d
                    weight_d = samp_weight * domain_d
                    if self.parameter == "ratio":
                        x_d = x * domain_d
                    else:
                        x_d = x
                    y_d = y * domain_d if len(y.shape) == 1 else y * domain_d[:, None]
                    y_score_d = self._score_variable(y_d, weight_d, x_d)
                    cov_score_d = self._taylor_variance(
                        y_score_d, weight_d, stratum, psu, ssu, fpc
                    )
                    variance3[d] = float(cov_score_d)
                return variance3, variance3

    def _estimate(
        self,
        y: Array,
        samp_weight: np.ndarray,
        x: Optional[np.ndarray],
        stratum: Optional[np.ndarray],
        psu: Optional[np.ndarray],
        ssu: Optional[np.ndarray],
        domain: Optional[np.ndarray],
        fpc: Union[Dict[StringNumber, Number], Number],
        deff: bool,
        coef_variation: bool,
        as_factor: bool,
        remove_nan: bool,
    ) -> None:

        self.point_est = self._get_point(
            y=y,
            samp_weight=samp_weight,
            x=x,
            domain=domain,
            as_factor=as_factor,
            remove_nan=remove_nan,
        )
        self.variance, self.covariance = self._get_variance(
            y=y,
            samp_weight=samp_weight,
            x=x,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            domain=domain,
            fpc=fpc,
            as_factor=as_factor,
            remove_nan=remove_nan,
        )

        self._degree_of_freedom(samp_weight, stratum, psu)
        t_quantile = student.ppf(1 - self.alpha / 2, df=self.degree_of_freedom)

        if domain is None:
            # breakpoint()
            if (
                (self.parameter == "proportion" or as_factor and self.parameter == "mean")
                and isinstance(self.point_est, dict)
                and isinstance(self.variance, dict)
            ):
                stderror: Dict[StringNumber, float] = {}
                lower_ci: Dict[StringNumber, float] = {}
                upper_ci: Dict[StringNumber, float] = {}
                coef_var: Dict[StringNumber, float] = {}
                # breakpoint()
                for level in self.variance:
                    point_est = self.point_est[level]
                    stderror[level] = math.sqrt(self.variance[level])
                    if point_est == 0:
                        lower_ci[level] = 0.0
                        upper_ci[level] = 0.0
                        coef_var[level] = 0.0
                    elif point_est == 1:
                        lower_ci[level] = 1.0
                        upper_ci[level] = 1.0
                        coef_var[level] = 0.0
                    elif isinstance(point_est, float):
                        location_ci = math.log(point_est / (1 - point_est))
                        scale_ci = stderror[level] / (point_est * (1 - point_est))
                        ll = location_ci - t_quantile * scale_ci
                        uu = location_ci + t_quantile * scale_ci
                        lower_ci[level] = math.exp(ll) / (1 + math.exp(ll))
                        upper_ci[level] = math.exp(uu) / (1 + math.exp(uu))
                        coef_var[level] = stderror[level] / point_est

                self.stderror = stderror
                self.coef_var = coef_var
                self.lower_ci = lower_ci
                self.upper_ci = upper_ci
            elif (
                as_factor and isinstance(self.point_est, dict) and isinstance(self.variance, dict)
            ):
                stderror = {}
                lower_ci = {}
                upper_ci = {}
                coef_var = {}
                # breakpoint()
                for level in self.variance:
                    stderror[level] = math.sqrt(self.variance[level])
                    lower_ci[level] = self.point_est[level] - t_quantile * stderror[level]
                    upper_ci[level] = self.point_est[level] + t_quantile * stderror[level]
                    coef_var[level] = stderror[level] / self.point_est[level]

                self.stderror = stderror
                self.coef_var = coef_var
                self.lower_ci = lower_ci
                self.upper_ci = upper_ci
            elif isinstance(self.variance, (int, float, np.ndarray)) and isinstance(
                self.point_est, (int, float, np.ndarray)
            ):
                self.stderror = math.sqrt(self.variance)
                self.lower_ci = self.point_est - t_quantile * self.stderror
                self.upper_ci = self.point_est + t_quantile * self.stderror
                self.coef_var = math.sqrt(self.variance) / self.point_est
        elif isinstance(self.variance, dict):
            for key in self.variance:
                if (
                    (self.parameter == "proportion" or as_factor and self.parameter == "mean")
                    and isinstance(self.point_est, dict)
                    and isinstance(self.variance, dict)
                ):
                    stderror = {}
                    lower_ci = {}
                    upper_ci = {}
                    coef_var = {}
                    for level in self.variance[key]:
                        point_est1 = self.point_est[key]
                        variance1 = self.variance[key]
                        if isinstance(point_est1, dict) and isinstance(variance1, dict):
                            stderror[level] = math.sqrt(variance1[level])
                            if point_est1[level] == 0:
                                lower_ci[level] = 0
                                upper_ci[level] = 0
                                coef_var[level] = 0
                            elif point_est1[level] == 1:
                                lower_ci[level] = 1
                                upper_ci[level] = 1
                                coef_var[level] = 0
                            else:
                                location_ci = math.log(point_est1[level] / (1 - point_est1[level]))
                                scale_ci = stderror[level] / (
                                    point_est1[level] * (1 - point_est1[level])
                                )
                                ll = location_ci - t_quantile * scale_ci
                                uu = location_ci + t_quantile * scale_ci
                                lower_ci[level] = math.exp(ll) / (1 + math.exp(ll))
                                upper_ci[level] = math.exp(uu) / (1 + math.exp(uu))
                                coef_var[level] = stderror[level] / point_est1[level]

                    self.stderror[key] = stderror
                    self.coef_var[key] = coef_var
                    self.lower_ci[key] = lower_ci
                    self.upper_ci[key] = upper_ci
                elif (
                    as_factor
                    and isinstance(self.point_est, dict)
                    and isinstance(self.variance, dict)
                ):
                    stderror = {}
                    lower_ci = {}
                    upper_ci = {}
                    coef_var = {}
                    # breakpoint()
                    for level in self.variance[key]:
                        stderror[level] = math.sqrt(self.variance[key][level])
                        lower_ci[level] = self.point_est[key][level] - t_quantile * stderror[level]
                        upper_ci[level] = self.point_est[key][level] + t_quantile * stderror[level]
                        coef_var[level] = stderror[level] / self.point_est[key][level]

                    self.stderror[key] = stderror
                    self.coef_var[key] = coef_var
                    self.lower_ci[key] = lower_ci
                    self.upper_ci[key] = upper_ci
                elif isinstance(self.point_est, dict):
                    self.stderror[key] = math.sqrt(self.variance[key])
                    self.lower_ci[key] = self.point_est[key] - t_quantile * self.stderror[key]
                    self.upper_ci[key] = self.point_est[key] + t_quantile * self.stderror[key]
                    self.coef_var[key] = math.sqrt(self.variance[key]) / self.point_est[key]

    def estimate(
        self,
        y: Array,
        samp_weight: Optional[Array] = None,
        x: Optional[Array] = None,
        stratum: Optional[Series] = None,
        psu: Optional[Series] = None,
        ssu: Optional[Series] = None,
        domain: Optional[Series] = None,
        by: Optional[Series] = None,
        fpc: Union[Dict[StringNumber, Number], Series, Number] = 1.0,
        deff: bool = False,
        coef_variation: bool = False,
        as_factor: bool = False,
        remove_nan: bool = False,
    ) -> None:

        if as_factor and self.parameter not in ("mean", "total"):
            raise AssertionError("When as_factor is True, parameter must be mean or total!")

        if self.parameter == "ratio" and x is None:
            raise AssertionError("x must be provided for ratio estimation.")

        y = numpy_array(y)
        y_temp = y.copy()

        if samp_weight is None:
            weight_temp = np.ones(y_temp.shape[0])
        elif isinstance(samp_weight, (float, int)):
            weight_temp = samp_weight * np.ones(y_temp.shape[0])
        elif isinstance(samp_weight, np.ndarray):
            weight_temp = samp_weight.copy()
        else:
            weight_temp = np.asarray(samp_weight)

        if remove_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y_temp) | np.isnan(x)
            else:
                excluded_units = np.isnan(y_temp)
            y_temp, weight_temp, x, stratum, psu, ssu, domain, by = remove_nans(
                excluded_units, y_temp, weight_temp, x, stratum, psu, ssu, domain, by
            )

        if stratum is not None:
            stratum = numpy_array(stratum)
            self.strata = np.unique(stratum).tolist()

        if domain is not None:
            domain = numpy_array(domain)
            self.domains = np.unique(domain).tolist()

        if by is not None:
            by = numpy_array(by)
            self.by = np.unique(by).tolist()

        if not isinstance(fpc, dict):
            self.fpc = fpc_as_dict(stratum, fpc)
        else:
            if list(np.unique(stratum)) != list(fpc.keys()):
                raise AssertionError("fpc dictionary keys must be the same as the strata!")
            else:
                self.fpc = fpc

        if by is None:
            self._estimate(
                y=y_temp,
                samp_weight=weight_temp,
                x=x,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                domain=domain,
                fpc=self.fpc,
                deff=deff,
                coef_variation=coef_variation,
                as_factor=as_factor,
                remove_nan=remove_nan,
            )
        else:
            for b in self.by:
                group_b = by == b
                # breakpoint()
                y_temp_b = y_temp[group_b]
                weight_temp_b = weight_temp[group_b]
                x_b = x[group_b] if x is not None else None
                stratum_b = stratum[group_b] if stratum is not None else None
                psu_b = psu[group_b] if psu is not None else None
                ssu_b = ssu[group_b] if ssu is not None else None
                domain_b = domain[group_b] if domain is not None else None

                by_est = TaylorEstimator(
                    parameter=self.parameter,
                    alpha=self.alpha,
                    random_seed=self.random_seed,
                    ciprop_method=self.ciprop_method,
                )
                by_est._estimate(
                    y=y_temp_b,
                    samp_weight=weight_temp_b,
                    x=x_b,
                    stratum=stratum_b,
                    psu=psu_b,
                    ssu=ssu_b,
                    domain=domain_b,
                    fpc=self.fpc,
                    deff=deff,
                    coef_variation=coef_variation,
                    as_factor=as_factor,
                    remove_nan=remove_nan,
                )

                self.point_est[b] = by_est.point_est
                self.stderror[b] = by_est.stderror
                self.lower_ci[b] = by_est.lower_ci
                self.upper_ci[b] = by_est.upper_ci
