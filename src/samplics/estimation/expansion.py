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
from __future__ import annotations

import math

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from scipy.stats import t as student

from samplics.utils.basic_functions import get_single_psu_strata
from samplics.utils.formats import dict_to_dataframe, fpc_as_dict, numpy_array, remove_nans
from samplics.utils.types import Array, Number, Series, StringNumber, SinglePSUEst, SamplicsError


class _SurveyEstimator:
    """General approach for sample estimation of linear parameters."""

    def __init__(self, parameter: str, alpha: float = 0.05, random_seed: Optional[int] = None):
        """Initializes the instance"""

        self.random_seed: Optional[int]
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

        self.point_est: Any = {}  # Union[dict[StringNumber, DictStrNum], DictStrNum, Number]
        self.variance: Any = {}
        self.covariance: Any = {}
        self.stderror: Any = {}
        self.coef_var: Any = {}
        self.deff: Any = {}
        self.lower_ci: Any = {}
        self.upper_ci: Any = {}
        self.fpc: Any = {}
        self.strata: Any = None
        self.single_psu_strata: Any = None
        self.domains: Any = None
        self.method: Any = "taylor"
        self.number_strata: Any = None
        self.number_psus: Any = None
        self.degree_of_freedom: Any = None
        self.as_factor: bool = False  # whether to treat outcome as factor.

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
            if self.domains is not None and (self.parameter == "proportion" or self.as_factor):
                domains = list()
                levels = list()
                point_est = list()
                stderror = list()
                lower_ci = list()
                upper_ci = list()
                coef_var = list()
                for domain in self.domains:
                    domains += np.repeat(domain, len(self.point_est[domain])).tolist()
                    levels += list(self.point_est[domain].keys())
                    point_est += list(self.point_est[domain].values())
                    stderror += list(self.stderror[domain].values())
                    lower_ci += list(self.lower_ci[domain].values())
                    upper_ci += list(self.upper_ci[domain].values())
                    coef_var += list(self.coef_var[domain].values())
                estimation["DOMAIN"] = domains
                estimation["LEVEL"] = levels
                estimation[parameter] = point_est
                estimation["SE"] = stderror
                estimation["LCI"] = lower_ci
                estimation["UCI"] = upper_ci
                estimation["CV"] = coef_var
            elif self.domains is not None:
                estimation["DOMAIN"] = self.domains
                estimation[parameter] = self.point_est.values()
                estimation["SE"] = self.stderror.values()
                estimation["LCI"] = self.lower_ci.values()
                estimation["UCI"] = self.upper_ci.values()
                estimation["CV"] = self.coef_var.values()
            else:
                estimation["LEVELS"] = self.point_est.keys()
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
        stratum: Optional[np.ndarray] = None,
        psu: Optional[np.ndarray] = None,
    ) -> None:

        stratum = numpy_array(stratum)
        psu = numpy_array(psu)

        if stratum.size <= 1:
            self.number_psus = np.unique(psu).size if psu.size > 1 else samp_weight.size
            self.number_strata = 1
        elif psu.size > 1:
            # if type object ("O") change it to str, otherwise np.unique will fail
            if stratum.dtype == "O":
                stratum = stratum.astype(str)
            if psu.dtype == "O":
                psu = psu.astype(str)
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
        elif self.parameter == "ratio" and x is not None:
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
    ) -> Any:  # Union[dict[StringNumber, DictStrNum], DictStrNum, Number]:
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
            if self.parameter == "ratio" and x is not None:
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
                cat_dict: dict[StringNumber, float] = {}
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
                estimate1: dict[StringNumber, dict[StringNumber, float]] = {}
                for d in domain_ids:
                    weight_d = samp_weight[domain == d]
                    cat_dict_d: dict[StringNumber, float] = {}
                    for k in range(categories.size):
                        y_d_k = y_dummies[domain == d, k]
                        cat_dict_d_k = dict({categories[k]: self._get_point_d(y_d_k, weight_d)})
                        cat_dict_d.update(cat_dict_d_k)
                    estimate1[d] = cat_dict_d
                return estimate1
            else:
                estimate2: dict[StringNumber, float] = {}
                for d in domain_ids:
                    weight_d = samp_weight[domain == d]
                    y_d = y[domain == d]
                    if x is not None:
                        x_d = x[domain == d] if self.parameter == "ratio" else None
                    else:
                        x_d = None
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
        """Initializes the instance"""
        _SurveyEstimator.__init__(self, parameter)
        if self.parameter == "proportion":
            self.ciprop_method = ciprop_method
        else:
            self.ciprop_method = None

    def _score_variable(
        self, y: np.ndarray, samp_weight: np.ndarray, x: Optional[np.ndarray] = None
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
            return np.asarray((y - location_weights) * samp_weight[:, None] / scale_weights)
        elif self.parameter == "ratio":
            weighted_sum_x = np.sum(x * samp_weight)
            weighted_ratio = np.sum(y_weighted, axis=0) / weighted_sum_x
            return np.asarray(
                samp_weight[:, None] * (y - x[:, None] * weighted_ratio) / weighted_sum_x
            )
        elif self.parameter == "total":
            return np.asarray(y_weighted)
        else:
            raise ValueError("parameter not valid!")

    @staticmethod
    def _variance_stratum_between(
        y_score_s: np.ndarray,
        samp_weight_s: np.ndarray,
        number_psus_in_s: int,
        psu_s: Optional[np.ndarray],
    ) -> np.ndarray:
        """Computes the variance for one stratum"""

        covariance = np.asarray([])
        if psu_s is not None:
            scores_s_mean = np.asarray(y_score_s.sum(axis=0) / number_psus_in_s)  # new
            psus = np.unique(psu_s)
            scores_psus_sums = np.zeros((number_psus_in_s, scores_s_mean.shape[0]))
            for k, psu in enumerate(np.unique(psus)):
                scores_psus_sums[k, :] = y_score_s[psu_s == psu].sum(axis=0)
            covariance = np.transpose(scores_psus_sums - scores_s_mean) @ (
                scores_psus_sums - scores_s_mean
            )
            covariance = (number_psus_in_s / (number_psus_in_s - 1)) * covariance
        else:
            number_obs = y_score_s.shape[0]
            y_score_s_mean = y_score_s.sum(axis=0) / number_obs
            covariance = (
                (number_obs / (number_obs - 1))
                * np.transpose(y_score_s - y_score_s_mean)
                @ (y_score_s - y_score_s_mean)
            )

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
        stratum: Optional[np.ndarray],
        psu: Optional[np.ndarray],
        ssu: Optional[np.ndarray] = None,
        fpc: Union[dict[StringNumber, Number], Number] = 1,
        skipped_strata: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes the variance across stratum"""

        if stratum is None and isinstance(fpc, (int, float)):
            number_psus = np.unique(psu).size
            return np.asarray(
                fpc
                * self._variance_stratum_between(
                    y_score_s=y_score,
                    samp_weight_s=samp_weight,
                    number_psus_in_s=number_psus,
                    psu_s=psu,
                )
            )
        elif isinstance(fpc, dict):
            covariance = np.zeros((y_score.shape[1], y_score.shape[1]))
            strata = np.unique(stratum)
            singletons = np.isin(strata, skipped_strata)
            for s in strata[~singletons]:
                y_score_s = y_score[stratum == s]
                samp_weight_s = samp_weight[stratum == s]
                psu_s = psu[stratum == s] if psu is not None else None  # np.array([])
                number_psus_in_s = np.size(np.unique(psu_s)) if psu_s is not None else 0
                ssu_s = ssu[stratum == s] if ssu is not None else None
                covariance += fpc[s] * self._variance_stratum_between(
                    y_score_s=y_score_s,
                    samp_weight_s=samp_weight_s,
                    number_psus_in_s=number_psus_in_s,
                    psu_s=psu_s,
                )
            return np.asarray(covariance)
        else:
            raise ValueError

    def _get_variance(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        x: Optional[np.ndarray] = None,
        stratum: Optional[np.ndarray] = None,
        psu: Optional[np.ndarray] = None,
        ssu: Optional[np.ndarray] = None,
        domain: Optional[np.ndarray] = None,
        fpc: Union[dict[StringNumber, Number], Number] = 1,
        skipped_strata: Optional[np.ndarray] = None,
        as_factor: bool = False,
        remove_nan: bool = False,
    ) -> tuple[
        Union[dict[StringNumber, Any], np.ndarray],
        Union[dict[StringNumber, Any], np.ndarray],
    ]:

        if self.parameter == "ratio" and x is None:
            raise AssertionError("Parameter x must be provided for ratio estimation.")

        if remove_nan:
            if self.parameter == "ratio" and x is not None:
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
                y, samp_weight, x, stratum, domain, psu, ssu = remove_nans(
                    excluded_units, y, samp_weight, x, stratum, domain, psu, ssu
                )

        if self.parameter == "proportion" or as_factor:
            y_df = pd.get_dummies(y).astype(int)
            categories = list(y_df.columns)
            y = y_df.values

        if domain is None:
            y_score = self._score_variable(y, samp_weight, x)  # new
            cov_score = self._taylor_variance(
                y_score=y_score,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
                skipped_strata=skipped_strata,
            )  # new
            if (self.parameter == "proportion" or as_factor) and isinstance(cov_score, np.ndarray):
                variance1: dict[StringNumber, dict] = {}
                covariance1: dict[StringNumber, dict] = {}
                variance1 = dict(zip(categories, np.diag(cov_score)))
                for k, level in enumerate(categories):
                    covariance1[level] = dict(zip(categories, cov_score[k, :]))
                return variance1, covariance1
            else:
                # Todo: generalize for multiple Y variables
                return float(cov_score), float(cov_score)
        else:
            domain = np.asarray(domain)
            if self.parameter == "proportion" or as_factor:
                variance2: dict[StringNumber, dict] = {}
                covariance2: dict[StringNumber, dict] = {}
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
                        y_score=y_score_d,
                        samp_weight=weight_d,
                        stratum=stratum,
                        psu=psu,
                        ssu=ssu,
                        fpc=fpc,
                        skipped_strata=skipped_strata,
                    )
                    variance2[d] = dict(zip(categories, np.diag(cov_score_d)))
                    cov_d = {}
                    for k, level in enumerate(categories):
                        cov_d.update({level: dict(zip(categories, cov_score_d[k, :]))})
                    covariance2.update({d: cov_d})
                return variance2, covariance2
            else:
                variance3: dict[StringNumber, float] = {}
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
                        y_score=y_score_d,
                        samp_weight=weight_d,
                        stratum=stratum,
                        psu=psu,
                        ssu=ssu,
                        fpc=fpc,
                        skipped_strata=skipped_strata,
                    )
                    variance3[d] = float(cov_score_d)
                return variance3, variance3

    def _estimate(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        x: Optional[np.ndarray],
        stratum: Optional[np.ndarray],
        psu: Optional[np.ndarray],
        ssu: Optional[np.ndarray],
        domain: Optional[np.ndarray],
        fpc: Union[dict[StringNumber, Number], Number],
        deff: bool,
        coef_variation: bool,
        skipped_strata: np.ndarray,
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
            skipped_strata=skipped_strata,
            as_factor=as_factor,
            remove_nan=remove_nan,
        )

        self._degree_of_freedom(samp_weight, stratum, psu)
        t_quantile = student.ppf(1 - self.alpha / 2, df=self.degree_of_freedom)

        if domain is None:
            if (
                (self.parameter == "proportion" or as_factor and self.parameter == "mean")
                and isinstance(self.point_est, dict)
                and isinstance(self.variance, dict)
            ):
                stderror: dict[StringNumber, float] = {}
                lower_ci: dict[StringNumber, float] = {}
                upper_ci: dict[StringNumber, float] = {}
                coef_var: dict[StringNumber, float] = {}
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

    def _raise_singleton_error(self):
        raise ValueError(f"Only one PSU in the following strata: {self.single_psu_strata}")

    def _skip_singleton(self, skipped_strata: Array) -> Array:
        skipped_str = np.isin(self.single_psu_strata, skipped_strata)
        if skipped_str.sum() > 0:
            return self.single_psu_strata[skipped_str]
        else:
            raise ValueError("{skipped_strata} does not contain singleton PSUs")

    @staticmethod
    def _certainty_singleton(
        singletons: Array,
        _stratum: Array,
        _psu: Array,
        _ssu: Array,
    ) -> Array:
        if _ssu is not None:
            certainties = np.isin(_stratum, singletons)
            _psu[certainties] = _ssu[certainties]
        else:
            for s in singletons:
                cert_s = np.isin(_stratum, s)
                nb_records = _psu[cert_s].shape[0]
                _psu[cert_s] = np.linspace(1, nb_records, num=nb_records, dtype="int")

        return _psu

    @staticmethod
    def _combine_strata(comb_strata: Array, _stratum: Array) -> Array:
        if comb_strata is None:
            raise ValueError("The parameter 'strata_comb' must be provided to combine strata")
        else:
            for s in comb_strata:
                _stratum[_stratum == s] = comb_strata[s]
            return _stratum

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
        fpc: Union[dict[StringNumber, Number], Series, Number] = 1.0,
        deff: bool = False,
        coef_variation: bool = False,
        single_psu: Union[SinglePSUEst, dict[StringNumber, SinglePSUEst]] = SinglePSUEst.error,
        strata_comb: Optional[dict[Array, Array]] = None,
        as_factor: bool = False,
        remove_nan: bool = False,
    ) -> None:

        if as_factor and self.parameter not in ("mean", "total"):
            raise AssertionError("When as_factor is True, parameter must be mean or total!")

        if self.parameter == "ratio" and x is None:
            raise AssertionError("x must be provided for ratio estimation.")

        y = numpy_array(y)
        y_temp = y.copy()

        x = numpy_array(x) if x is not None else None
        _stratum = numpy_array(stratum).copy() if stratum is not None else None
        _psu = numpy_array(psu).copy() if psu is not None else None
        _ssu = numpy_array(ssu).copy() if ssu is not None else None

        if samp_weight is None:
            weight_temp = np.ones(y_temp.shape[0])
        elif isinstance(samp_weight, (float, int)):
            weight_temp = samp_weight * np.ones(y_temp.shape[0])
        elif isinstance(samp_weight, np.ndarray):
            weight_temp = samp_weight.copy()
        else:
            weight_temp = np.asarray(samp_weight)

        if remove_nan:
            if self.parameter == "ratio" and x is not None:
                excluded_units = np.isnan(y_temp) | np.isnan(x)
            else:
                excluded_units = np.isnan(y_temp)
            y_temp, weight_temp, x, _stratum, _psu, _ssu, domain, by = remove_nans(
                excluded_units, y_temp, weight_temp, x, _stratum, _psu, _ssu, domain, by
            )

        if _stratum is not None:
            # _stratum = numpy_array(stratum).copy()
            self.strata = np.unique(_stratum).tolist()
            # TODO: we could improve efficiency by creating the pair [stratum,psu, ssu] ounce and
            # use it in get_single_psu_strata and in the uncertainty calculation functions
            self.single_psu_strata = get_single_psu_strata(_stratum, _psu)

        skipped_strata = None
        if self.single_psu_strata is not None:
            if single_psu == SinglePSUEst.error:
                self._raise_singleton_error()
            if single_psu == SinglePSUEst.skip:
                skipped_strata = self._skip_singleton(skipped_strata=self.single_psu_strata)
            if single_psu == SinglePSUEst.certainty:
                _psu = self._certainty_singleton(
                    singletons=self.single_psu_strata, _stratum=_stratum, _psu=_psu, _ssu=_ssu
                )
            if single_psu == SinglePSUEst.combine:
                _stratum = self._combine_strata(strata_comb, _stratum)
            # TODO: more method for singleton psus to be implemented
            if isinstance(single_psu, dict):
                for s in single_psu:
                    if single_psu[s] == SinglePSUEst.error:
                        self._raise_singleton_error()
                    if single_psu[s] == SinglePSUEst.skip:
                        skipped_strata = self._skip_singleton(skipped_strata=numpy_array(s))
                    if single_psu[s] == SinglePSUEst.certainty:
                        _psu = self._certainty_singleton(
                            singletons=numpy_array(s), _stratum=_stratum, _psu=_psu, _ssu=_ssu
                        )
                    if single_psu[s] == SinglePSUEst.combine:
                        _stratum = self._combine_strata(strata_comb, _stratum)

            skipped_strata = get_single_psu_strata(_stratum, _psu)
            if skipped_strata is not None and single_psu in [
                SinglePSUEst.certainty,
                SinglePSUEst.combine,
            ]:  # TODO: add the left our singletons when using the dict instead of SinglePSUEst
                self._raise_singleton_error()

        if domain is not None:
            domain = numpy_array(domain)
            self.domains = np.unique(domain).tolist()

        if by is not None:
            by = numpy_array(by)
            self.by = np.unique(by).tolist()

        if not isinstance(fpc, dict):
            self.fpc = fpc_as_dict(stratum, fpc)
        else:
            if list(np.unique(_stratum)) != list(fpc.keys()):
                raise AssertionError("fpc dictionary keys must be the same as the strata!")
            else:
                self.fpc = fpc

        self.as_factor = as_factor

        if by is None:
            self._estimate(
                y=y_temp,
                samp_weight=weight_temp,
                x=x,
                stratum=_stratum,
                psu=_psu,
                ssu=_ssu,
                domain=domain,
                fpc=self.fpc,
                deff=deff,
                coef_variation=coef_variation,
                skipped_strata=skipped_strata,
                as_factor=as_factor,
                remove_nan=remove_nan,
            )
        else:
            for b in self.by:
                group_b = by == b
                y_temp_b = y_temp[group_b]
                weight_temp_b = weight_temp[group_b]
                x_b = x[group_b] if x is not None else None
                _stratum_b = _stratum[group_b] if _stratum is not None else None
                _psu_b = _psu[group_b] if _psu is not None else None
                _ssu_b = _ssu[group_b] if _ssu is not None else None
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
                    stratum=_stratum_b,
                    psu=_psu_b,
                    ssu=_ssu_b,
                    domain=domain_b,
                    fpc=self.fpc,
                    deff=deff,
                    coef_variation=coef_variation,
                    skipped_strata=skipped_strata,
                    as_factor=as_factor,
                    remove_nan=remove_nan,
                )

                self.point_est[b] = by_est.point_est
                self.stderror[b] = by_est.stderror
                self.lower_ci[b] = by_est.lower_ci
                self.upper_ci[b] = by_est.upper_ci

    def to_dataframe(
        self,
        col_names: Optional(list) = None,
    ) -> pd.DataFrame:

        if self.point_est is None:
            raise AssertionError("No estimates yet. Must first run estimate().")
        elif col_names is None:
            if self.parameter == "proportion" or self.as_factor:
                col_names = [
                    "_parameter",
                    "_domain",
                    "_level",
                    "_estimate",
                    "_stderror",
                    "_lci",
                    "_uci",
                    "_cv",
                    "_deff",
                ]
            else:
                col_names = [
                    "_parameter",
                    "_domain",
                    "_estimate",
                    "_stderror",
                    "_lci",
                    "_uci",
                    "_cv",
                    "_deff",
                ]
            if self.deff == {}:
                col_names.pop()
            if self.domains is None:
                col_names.pop(1)
        else:
            ncols = len(col_names)
            if self.deff is not None and self.as_factor is not None and ncols != 9:
                raise AssertionError("col_names must have 9 values")
            if self.deff is None and self.as_factor is not None and ncols != 8:
                raise AssertionError("col_names must have 8 values")
            if self.deff is not None and self.as_factor is None and ncols != 8:
                raise AssertionError("col_names must have 8 values")
            if self.deff is None and self.as_factor is None and ncols != 7:
                raise AssertionError("col_names must have 7 values")

        if self.deff == {}:
            est_df = dict_to_dataframe(
                col_names,
                self.point_est,
                self.stderror,
                self.lower_ci,
                self.upper_ci,
                self.coef_var,
            )
        else:
            est_df = dict_to_dataframe(
                col_names,
                self.point_est,
                self.stderror,
                self.lower_ci,
                self.upper_ci,
                self.coef_var,
                self.deff,
            )
        est_df.iloc[:, 0] = self.parameter

        return est_df
