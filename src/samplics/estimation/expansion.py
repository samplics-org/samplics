"""Estimation of linear parameters
"""

import numpy as np
import pandas as pd

from scipy.stats import norm as normal
from scipy.stats import t as student

import math

from samplics.utils import checks, formats


class _SurveyEstimator:
    """ General approach for sample estimation of linear parameters
    """

    def __init__(self, parameter, alpha=0.05, random_seed=None):
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

        self.point_est = {}
        self.variance = {}
        self.stderror = {}
        self.coef_var = {}
        self.deff = {}
        self.lower_ci = {}
        self.upper_ci = {}
        self.strata = "__none__"
        self.domains = "__none__"
        self.method = "taylor"
        self.number_strata = None
        self.number_psus = None
        self.degree_of_freedom = None
        self.alpha = alpha
        self.number_reps = None
        self.rep_coefs = None

    def __str__(self):
        parameter = self.parameter.upper()
        estimation = pd.DataFrame()
        estimation["DOMAINS"] = self.domains
        estimation[parameter] = self.point_est.values()
        estimation["SE"] = math.pow(self.variance.values(), 0.5)
        estimation["LCI"] = self.lower_ci.values()
        estimation["UCI"] = self.upper_ci.values()
        estimation["CV"] = self.coef_var.values()

        return "%s" % estimation

    def __repr__(self):
        return self.__str__()

    def _exclude_nans(
        self,
        excluded_units,
        y,
        sample_weight,
        x=None,
        stratum=None,
        domain=None,
        psu=None,
        ssu=None,
    ):
        y = formats.numpy_array(y)
        sample_weight = formats.numpy_array(sample_weight)
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
            sample_weight[~excluded_units],
            x,
            stratum,
            domain,
            psu,
            ssu,
        )

    def _degree_of_freedom(self, sample_weight, stratum=None, psu=None):

        stratum = formats.numpy_array(stratum)
        psu = formats.numpy_array(psu)

        if stratum.size <= 1:
            self.degree_of_freedom = np.unique(psu).size - 1
        elif psu.size > 1:
            self.degree_of_freedom = np.unique(psu).size - np.unique(stratum).size
        else:
            sample_weight = formats.numpy_array(sample_weight)
            self.degree_of_freedom = sample_weight.size

    def _get_point(self, y, sample_weight, x=None):

        if self.parameter in ("proportion", "mean"):
            return np.sum(sample_weight * y) / np.sum(sample_weight)
        elif self.parameter == "total":
            return np.sum(sample_weight * y)
        elif self.parameter == "ratio":
            return np.sum(sample_weight * y) / np.sum(sample_weight * x)

    def get_point(self, y, sample_weight, x=None, domain=None, exclude_nan=False):
        """Computes the parameter point estimates

        Args:

        y:

        sample_weight:

        domain:

        Returns:
        A float or dictionary: The estimated parameter

        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("Parameter x must be provided for ratio estimation.")

        if exclude_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, sample_weight, x, _, domain, _, _ = self._exclude_nans(
                excluded_units, y, sample_weight, x, domain=domain
            )

        if self.parameter == "proportion":
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values

        estimate = {}
        if domain is None:
            if self.parameter == "proportion":
                cat_dict = dict()
                for k in range(categories.size):
                    y_k = y_dummies[:, k]
                    cat_dict_k = dict({categories[k]: self._get_point(y_k, sample_weight)})
                    cat_dict.update(cat_dict_k)
                estimate["__none__"] = cat_dict
            else:
                estimate["__none__"] = self._get_point(y, sample_weight, x)
        else:
            domain_ids = np.unique(domain)
            for d in domain_ids:
                weight_d = sample_weight[domain == d]
                if self.parameter == "proportion":
                    cat_dict = dict()
                    for k in range(categories.size):
                        y_d_k = y_dummies[domain == d, k]
                        cat_dict_d_k = dict({categories[k]: self._get_point(y_d_k, weight_d)})
                        cat_dict.update(cat_dict_d_k)
                    estimate[d] = cat_dict
                else:
                    y_d = y[domain == d]
                    x_d = x[domain == d] if self.parameter == "ratio" else None
                    estimate[d] = self._get_point(y_d, weight_d, x_d)

        return estimate


class TaylorEstimator(_SurveyEstimator):
    """
    Taylor method to estimate uncertainty
    """

    def __init__(self, parameter, alpha=0.05, random_seed=None, ciprop_method="logit"):
        """Initializes the instance """
        _SurveyEstimator.__init__(self, parameter)
        if self.parameter == "proportion":
            self.ciprop_method = ciprop_method

    def _score_variable(self, y, sample_weight, x=None):
        """Provides the scores used to calculate the variance
        """

        y_weighted = y * sample_weight
        if self.parameter in ("proportion", "mean"):
            scale_weights = np.sum(sample_weight)
            location_weights = np.sum(y_weighted) / scale_weights
            return sample_weight * (y - location_weights) / scale_weights
        elif self.parameter == "ratio":
            weighted_sum_x = np.sum(x * sample_weight)
            weighted_ratio = np.sum(y_weighted) / weighted_sum_x
            return sample_weight * (y - x * weighted_ratio) / weighted_sum_x
        else:  # self.parameter == "total":
            return y_weighted

    @staticmethod
    def _variance_stratum_between(y_score_s, number_psus_in_s, psus_s):
        """Computes the variance for one stratum """

        scores_s_mean = y_score_s.sum() / number_psus_in_s
        variance = 0.0
        if number_psus_in_s > 1:
            psus = np.unique(psus_s)
            scores_psus_sums = np.zeros(number_psus_in_s)
            for k, psu in enumerate(np.unique(psus)):
                scores_psus_sums[k] = y_score_s[psus_s == psu].sum()

                variance = ((scores_psus_sums - scores_s_mean) ** 2).sum()
                variance = (number_psus_in_s / (number_psus_in_s - 1)) * variance

        return variance

    @staticmethod
    def _variance_stratum_within(y_score_s, number_psus_in_s, psu_s, ssu_s):

        variance = 0.0

        if ssu_s != None:
            psus = np.unique(psu_s)
            for psu in np.unique(psus):
                scores_psu_mean = y_score_s[psus == psu].mean()
                ssus = np.unique(ssu_s[psu_s == psu])
                number_ssus_in_psu = np.size(ssus)
                scores_ssus_sums = np.zeros(number_ssus_in_psu)
                if number_ssus_in_psu > 1:
                    for ssu in np.unique(ssus):
                        scores_ssus_sums[k] = y_score_s[ssu_s == ssu].sum()
                    variance += (number_ssus_in_psu / (number_ssus_in_psu - 1)) * (
                        (scores_ssus_sums - scores_psu_mean) ** 2
                    ).sum()

        return variance

    def _taylor_variance(self, y_score, stratum, psu, ssu=None):
        """Computes the variance across stratum """

        if stratum is None:
            number_psus = np.unique(psu).size
            var_est = self._variance_stratum_between(
                y_score, number_psus, psu
            ) + self._variance_stratum_within(y_score, number_psus, psu, ssu)

        else:
            var_est = 0.0
            for s in np.unique(stratum):
                y_score_s = y_score[stratum == s]
                psu_s = psu[stratum == s]
                number_psus_in_s = np.size(np.unique(psu_s))

                if ssu is not None:
                    ssu_s = ssus[stratum == s]
                else:
                    ssu_s = None

                var_est += self._variance_stratum_between(
                    y_score_s, number_psus_in_s, psu_s
                ) + self._variance_stratum_within(y_score_s, number_psus_in_s, psu_s, ssu_s)

        return var_est

    def get_variance(
        self,
        y,
        sample_weight,
        x=None,
        stratum=None,
        psu=None,
        ssu=None,
        domain=None,
        exclude_nan=False,
    ):
        """Computes the variance

        Args:

        y:

        sample_weight:

        domains:

        Returns:
        A dictionary

        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("Parameter x must be provided for ratio estimation.")

        if exclude_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, sample_weight, x, stratum, domain, psu, ssu = self._exclude_nans(
                excluded_units, y, sample_weight, x, stratum, domain, psu, ssu
            )

        if self.parameter == "proportion":
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values

        variance = {}
        if domain is None:
            if self.parameter == "proportion":
                cat_dict = dict()
                for k in range(categories.size):
                    y_score_k = self._score_variable(y_dummies[:, k], sample_weight)
                    cat_dict_k = dict(
                        {categories[k]: self._taylor_variance(y_score_k, stratum, psu, ssu)}
                    )
                    cat_dict.update(cat_dict_k)
                variance["__none__"] = cat_dict
            else:
                y_score = self._score_variable(y, sample_weight, x)
                variance[self.domains] = self._taylor_variance(y_score, stratum, psu, ssu)

        else:
            for d in np.unique(domain):
                weight_d = sample_weight * (domain == d)
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
                            {categories[k]: self._taylor_variance(y_score_d_k, stratum, psu, ssu)}
                        )
                        cat_dict.update(cat_dict_d_k)
                    variance[d] = cat_dict
                else:
                    y_d = y * (domain == d)
                    y_score_d = self._score_variable(y_d, weight_d, x_d)
                    variance[d] = self._taylor_variance(y_score_d, stratum, psu, ssu)

        return variance

    def get_confint(
        self,
        y,
        sample_weight,
        x=None,
        stratum=None,
        psu=None,
        ssu=None,
        domain=None,
        exclude_nan=False,
    ):
        """Computes the confidence interval

        Args:

        y:

        sample_weight:

        domain:

        Returns:
        A dictionary: Each dictionary value is a tupple (lower_ci, upper_ci)

        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("Parameter x must be provided for ratio estimation.")

        if exclude_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, sample_weight, x, stratum, domain, psu, ssu = self._exclude_nans(
                excluded_units, y, sample_weight, x, stratum, domain, psu, ssu
            )

        self._degree_of_freedom(sample_weight, stratum, psu)
        t_quantile = student.ppf(1 - self.alpha / 2, df=self.degree_of_freedom)
        # t_quantile = normal.ppf(1 - self.alpha / 2)

        estimate = self.get_point(y, sample_weight, x, domain)
        variance = self.get_variance(y, sample_weight, x, stratum, psu, ssu, domain)

        lower_ci = {}
        upper_ci = {}
        for key in variance:
            if self.parameter == "proportion":
                lower_ci_k = {}
                upper_ci_k = {}
                for level in variance[key]:
                    point_est = estimate[key][level]
                    std_est = pow(variance[key][level], 0.5)
                    location_ci = math.log(point_est / (1 - point_est))
                    scale_ci = std_est / (point_est * (1 - point_est))
                    ll = location_ci - t_quantile * scale_ci
                    lower_ci_k[level] = math.exp(ll) / (1 + math.exp(ll))
                    uu = location_ci + t_quantile * scale_ci
                    upper_ci_k[level] = math.exp(uu) / (1 + math.exp(uu))
                lower_ci[key] = lower_ci_k
                upper_ci[key] = upper_ci_k
            else:
                lower_ci[key] = estimate[key] - t_quantile * pow(variance[key], 0.5)
                upper_ci[key] = estimate[key] + t_quantile * pow(variance[key], 0.5)

        return lower_ci, upper_ci

    def get_coefvar(
        self,
        y,
        sample_weight,
        x=None,
        stratum=None,
        psu=None,
        ssu=None,
        domain=None,
        exclude_nan=False,
    ):
        """Computes the coefficient of variation

        Args:

        y:

        sample_weight:

        domain:

        Returns:
        A float or dictionnary: 

        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("Parameter x must be provided for ratio estimation.")

        if exclude_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, sample_weight, x, stratum, domain, psu, ssu = self._exclude_nans(
                excluded_units, y, sample_weight, x, stratum, domain, psu, ssu
            )

        estimate = self.get_point(y, sample_weight, x, domain)
        variance = self.get_variance(y, sample_weight, x, stratum, psu, ssu, domain)

        coef_var = {}
        for key in variance:
            if self.parameter == "proportion":
                coef_var_k = {}
                for level in variance[key]:
                    coef_var_k[level] = pow(variance[key][level], 0.5) / estimate[key][level]
                coef_var[key] = coef_var_k
            else:
                coef_var[key] = pow(variance[key], 0.5) / estimate[key]

        return coef_var

    def estimate(
        self,
        y,
        sample_weight,
        x=None,
        stratum=None,
        psu=None,
        ssu=None,
        domain=None,
        deff=False,
        coef_variation=False,
        exclude_nan=False,
    ):
        """Computes the parameter point estimates

        Args:

        y:

        weights:

        domains:

        Returns:
        A tupple (estimate, variance or stderror, coef_var?, lower_ci, upper_ci)

        """

        if self.parameter == "ratio" and x is None:
            raise AssertionError("x must be provided for ratio estimation.")

        if exclude_nan:
            if self.parameter == "ratio":
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, sample_weight, x, stratum, domain, psu, ssu = self._exclude_nans(
                excluded_units, y, sample_weight, x, stratum, domain, psu, ssu
            )

        if stratum is not None:
            self.strata = np.unique(stratum).tolist()
        if domain is not None:
            self.domains = np.unique(domain).tolist()

        self.point_est = self.get_point(y, sample_weight, x, domain)
        self.variance = self.get_variance(y, sample_weight, x, stratum, psu, ssu, domain)

        self._degree_of_freedom(sample_weight, stratum, psu)
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
                    location_ci = math.log(point_est / (1 - point_est))
                    scale_ci = stderror[level] / (point_est * (1 - point_est))
                    ll = location_ci - t_quantile * scale_ci
                    lower_ci[level] = math.exp(ll) / (1 + math.exp(ll))
                    uu = location_ci + t_quantile * scale_ci
                    upper_ci[level] = math.exp(uu) / (1 + math.exp(uu))
                    coef_var[level] = stderror[level] / point_est

                self.stderror[key] = stderror
                self.coef_var[key] = coef_var
                self.lower_ci[key] = lower_ci
                self.upper_ci[key] = upper_ci
            else:
                self.stderror[key] = pow(self.variance[key], 0.5)
                self.lower_ci[key] = self.point_est[key] - t_quantile * pow(
                    self.variance[key], 0.5
                )
                self.upper_ci[key] = self.point_est[key] + t_quantile * pow(
                    self.variance[key], 0.5
                )
                self.coef_var[key] = pow(self.variance[key], 0.5) / self.point_est[key]

        return self
