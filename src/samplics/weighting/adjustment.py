"""
Author: Mamadou S Diallo <msdiallo@QuantifyAfrica.org>


License: MIT
"""

from typing import Optional, Any, Union, Dict, Tuple

import numpy as np
import pandas as pd

from samplics.utils import checks, formats


class SampleWeight:
    """Sample weight 
    """

    def __init__(self) -> None:

        self.number_domains = 1
        self.number_units: Dict = {}
        self.deff_wgt: Dict = {}
        self.normalize_factor: Dict = {}
        self.normalize_level: Dict = {}
        self.adjustment_factor: Dict = {}
        self.triming_method = ""
        self.calibration_method = ""

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    def _number_units(self, domain: np.ndarray, sample_weight: np.ndarray) -> None:
        """Returns the number of units"""

        if domain in (None, dict()):
            self.number_units["__none__"] = len(sample_weight)
        elif domain is not None:
            keys, values = np.unique(domain, return_counts=True)
            self.number_units = dict(zip(keys, values))

    @staticmethod
    def _deff_wgt(sample_weight: np.ndarray) -> float:
        """compute the design effect due to unequal weights -
        Page 71 of Valliant and Dever (2018) """

        mean_w = np.mean(sample_weight)
        relvar_w = np.power(sample_weight - mean_w, 2) / mean_w ** 2
        deff_w = 1 + np.mean(relvar_w)

        return deff_w

    def deff_weight(self, sample_weight: np.ndarray, domain: np.ndarray = None) -> Dict:
        """compute the design effect due to unequal weights across estimation domains.  

        Args:
            sample_weight (array) : Array of the pre-adjsutment sample 
            weight. This vector should contains numeric values.   

            domain (array) : Array indicating the domains of interest 
            for the calculation of the design effect due to weighting.

        
        Returns:
            A dictionary: the mapping of the domains with associated 
            design effect due to weighting.
        """

        deff_w = {}
        if domain is None:
            deff_w["__none__"] = self._deff_wgt(sample_weight)
        else:
            for d in np.unique(domain):
                deff_w[d] = self._deff_wgt(sample_weight[domain == d])
        self.deff_wgt = deff_w

        return deff_w

    def extreme(self):

        pass

    def plot(self):

        pass

    @staticmethod
    def _norm_adjustment(
        sample_weight: np.ndarray, norm_level: Union[float, Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:

        sum_weights = np.sum(sample_weight)
        norm_factor = norm_level / sum_weights
        normalized_weight = sample_weight * norm_factor

        return normalized_weight, norm_factor

    @staticmethod
    def _response(response_status: np.ndarray, response_dict: np.ndarray) -> np.ndarray:

        response_status = formats.numpy_array(response_status)
        checks.check_response_status(response_status, response_dict)

        if not np.isin(response_status, ("in", "rr", "nr", "uk")).any():
            response_code = np.repeat("  ", response_status.size).astype(str)
            response_code[response_status == response_dict["in"]] = "in"
            response_code[response_status == response_dict["rr"]] = "rr"
            response_code[response_status == response_dict["nr"]] = "nr"
            response_code[response_status == response_dict["uk"]] = "uk"
        else:
            response_code = response_status

        return response_code

    @staticmethod
    def _adjustment_factor(
        sample_weight: np.ndarray, response_code: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        in_sample = response_code == "in"  # ineligible
        rr_sample = response_code == "rr"  # respondent
        nr_sample = response_code == "nr"  # nonrespondent
        uk_sample = response_code == "uk"  # unknown

        in_weights_sum = np.sum(sample_weight[in_sample])
        rr_weights_sum = np.sum(sample_weight[rr_sample])
        nr_weights_sum = np.sum(sample_weight[nr_sample])
        uk_weights_sum = np.sum(sample_weight[uk_sample])

        adjustment_rr = (rr_weights_sum + nr_weights_sum + uk_weights_sum) / rr_weights_sum

        adjustment_factor = np.ones(sample_weight.size)  # ineligibles will get 1 by default
        adjustment_factor[rr_sample] = adjustment_rr
        adjustment_factor[nr_sample] = 0
        adjustment_factor[uk_sample] = 0

        return adjustment_factor, adjustment_rr

    def adjust(
        self,
        sample_weight: np.ndarray,
        adjustment_class: np.ndarray,
        response_status: np.ndarray,
        response_dict: Dict = None,
    ) -> np.ndarray:
        """
        adjust sample weight to account for non-response. 

        Args:
            sample_weight (array) : Array of the pre-adjsutment sample 
            weight. This vector should contains numeric values.   

            adjustment_class (array) : Array indicating the adjustment 
            class for each sample unit. The sample weight adjustments 
            will be performed within the classes defined by this 
            parameter.

            response_status (array) : Array indicating the eligibility 
            and response status of the sample unit. Values of 
            response_status should inform on ineligible (in), 
            respondent (rr), nonrespondent (nr), 
            not known / unknown (uk). If the values of the paramter 
            are not in ("in", "rr", "nr", "uk") then the response_dict 
            is required.

            response_dict (dictionary) : Dictionnary providing the 
            mapping between the values of response_status and the 
            ["in", "rr", "nr", "uk"]. For example, if the response 
            status are: 0 for ineligible, 1 for respondent, 
            2 for nonrespondent, and 9 for unknown. Then the dictionary 
            will be {"in": 0, "rr": 1, "nr": 2, "uk": 9}. 
            If the response status variable has only values in 
            ("in", "rr", "nr", "uk") then the dictionary is not needed. 
            Optional parameter.

        
        Returns:
            An array: the adjsuted sample weight.
        """

        response_code = self._response(response_status, response_dict)
        sample_weight = formats.numpy_array(sample_weight)
        adjustment_class = formats.non_missing_array(adjustment_class)
        adjusted_weight = np.ones(sample_weight.size) * np.nan

        if adjustment_class.size <= 1:
            adjust_factor, self.adjustment_factor["__none__"] = self._adjustment_factor(
                sample_weight, response_code
            )
            adjusted_weight = adjust_factor * sample_weight
        else:
            for c in np.unique(adjustment_class):
                sample_weight_c = sample_weight[adjustment_class == c]
                response_code_c = response_code[adjustment_class == c]
                adjust_factor_c, self.adjustment_factor[c] = self._adjustment_factor(
                    sample_weight_c, response_code_c
                )
                adjusted_weight[adjustment_class == c] = adjust_factor_c * sample_weight_c
        self.deff_wgt = self.deff_weight(adjusted_weight)

        return adjusted_weight

    def poststratify(
        self,
        sample_weight: np.ndarray,
        x: np.ndarray,
        control_totals: Union[float, Dict[float, float]],
        constant_factor=1,
        domain=None,
    ) -> np.ndarray:

        return self.calibrate(sample_weight, X, control_totals, constant_factor, domain)

    @staticmethod
    def _core_matrix_d(
        sample_weight_d: np.ndarray,
        x_d: np.ndarray,
        x_weighted_totals_d: np.ndarray,
        control_totals_d: np.ndarray,
        constants_d=1,
    ):

        if np.size(constants_d) == 1:
            vect_ones = np.ones(np.size(sample_weight_d))
            v = constants_d * np.diag(vect_ones)
        else:
            v = np.diag(constants_d)

        core_matrix = np.transpose(x_d) * np.diag(sample_weight_d) * np.inv(v) * X_d
        core_matrix_inv = np.inv(core_matrix)
        core_factor = np.transpose(control_totals_d - x_weighted_totals_d) * core_matrix_inv

        return core_factor

    @staticmethod
    def _core_vector_d(x_i, core_factor):

        return np.dot(core_factor, x_i)

    def _poststratified_wgt(
        self,
        sample_weight: np.ndarray,
        X: np.ndarray,
        control_totals: Dict[Any, float],
        domain: np.ndarray = None,
    ):

        pass

    def _raked_wgt(self, sample_weight, X, control_totals, domain):
        pass

    def _calibrated_wgt(
        self,
        sample_weight: np.ndarray,
        X: np.ndarray,
        control_totals: Dict[Any, float],
        constant_factor: float,
        domain: np.ndarray,
    ) -> np.ndarray:
        x_weighted_total = np.dot(X, sample_weight)
        if domain is None:
            core_factor = self._core_matrix_d(
                sample_weight_d=sample_weight,
                x_d=X,
                x_weighted_totals_d=x_weighted_total,
                control_totals_d=control_totals,
                constants_d=constant_factor,
            )
            adjustment_factor = np.apply_along_axis(
                self._core_vector_d, axis=0, core_factor=core_factor
            )
            adjusted_factor = 1 + adjustment_factor / constant_factor
        else:
            adjusted_factor = []
            for d in np.unique(domain):
                x_d = X[domain == d]
                sample_weight_d = X[domain == d]
                x_weighted_total_d = x_weighted_total[domain == d]
                control_totals_d = control_totals[domain == d]
                constant_factor_d = control_totals[domain == d]
                core_factor_d = self._core_matrix_d(
                    sample_weight_d=sample_weight_d,
                    x_d=x_d,
                    x_weighted_totals_d=X_weighted_total_d,
                    control_totals_d=control_totals_d,
                    constants_d=constant_factor_d,
                )
                adjustment_factor_d = np.apply_along_axis(
                    self._core_vector_d, axis=0, core_factor=core_factor_d
                )
                adjusted_factor_d = 1 + adjustment_factor_d / constant_factor_d
                adjusted_factor = np.append(adjusted_factor, adjusted_factor_d)

        return adjusted_factor * sample_weight, adjustment_factor

    def calibrate(
        self,
        sample_weight: np.ndarray,
        x: np.ndarray,
        control_totals: Any,
        constant_factor=1,
        domains=None,
    ) -> np.ndarray:

        if self.calibration_method == "poststratification":
            return self._poststratified_wgt(sample_weight, x, control_totals, domain)
        elif self.calibration_method == "raking":
            return self._raked_wgt(sample_weight, x, control_totals, domain)
        elif self.calibration_method == "calibration":
            return self._calibrated_wgt(sample_weight, x, control_totals, constant_factor, domain)

    def trim(
        self,
        sample_weight: np.ndarray,
        triming_method: str,
        triming_class: np.ndarray,
        triming_level: Union[float, Dict[float, float]],
    ) -> np.ndarray:
        """
        trim sample weight to reduce the influence of extreme weights. 

        Args:
            sample_weight (array) : Array of the pre-adjsutment sample 
            weight. This vector should contains numeric values.   

            trim_method (string) : Name of the trimming method. 
            Possible values are: "threshold", "interquartile", and "?"

            trim_class (array) : Array indicating the trimming class 
            for each sample unit.

            trim_level (int, float, dictionary) : A number defining 
            the threshold or a dictionnary mapping triming classes 
            to trimming thresholds. Depending on the trimming method, 
            this parameter may provide the 

        
        Returns:
            An array: the trimmed sample weight.
        """

        pass

    def normalize(
        self,
        sample_weight: np.ndarray,
        norm_class: np.ndarray = None,
        norm_level: Union[float, Dict[float, float]] = None,
    ) -> np.ndarray:
        """
        normalize the sample weights to sum to a known constants or 
        levels. 

        Args:
            sample_weight (array) : Array of the pre-adjsutment sample 
            weight. This vector should contains numeric values.   

            norm_class (array) : Array indicating the normalization 
            class for each sample unit.

            norm_level (int, float, dictionary) : A number or array of 
            the level to calibrate the sum of the weights. Default is 
            number of units by norm_class key or overall if 
            norm_class is None. 

        
        Returns:
            An arrays: the normalized sample weight.
        """

        sample_weight = formats.numpy_array(sample_weight)
        norm_weight = sample_weight.copy()

        if norm_class is not None:
            norm_class = formats.numpy_array(norm_class)
            keys = np.unique(norm_class)
            levels: np.ndarray = np.zeros(keys.size) * np.nan
            for k, key in enumerate(keys):
                weight_k = sample_weight[norm_class == key]
                if norm_level is None:
                    levels[k] = np.sum(norm_class == key)
                elif norm_level is not None and not isinstance(norm_level, (float, int)):
                    levels[k] = norm_level[key]
                elif isinstance(norm_level, (float, int)):
                    levels[k] = norm_level
                (
                    norm_weight[norm_class == key],
                    self.normalize_factor[key],
                ) = self._norm_adjustment(weight_k, levels[k])
                self.normalize_level[key] = levels[k]
        else:
            if norm_level is not None:
                level = norm_level
            else:
                level = np.sum(sample_weight.size)
            norm_weight, self.normalize_factor["__none__"] = self._norm_adjustment(
                sample_weight, level
            )
            self.normalize_level["__none__"] = level

        return norm_weight
