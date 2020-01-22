"""
Author: Mamadou S Diallo <msdiallo@QuantifyAfrica.org>


License: MIT
"""

from typing import Optional, Any, Union, Dict, Tuple, List

import numpy as np
import pandas as pd

from samplics.utils import checks, formats


class SampleWeight:
    """Sample weight 
    """

    def __init__(self) -> None:

        self.number_scale_d = 1
        self.number_units: Dict = {}
        self.deff_wgt: Dict = {}
        self.norm_factor: Dict = {}
        self.control: Dict = {}
        self.adjust_factor: Dict = {}
        self.trim_method = ""
        self.calib_factor = ""

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    def _number_units(self, domain: np.ndarray, samp_weight: np.ndarray) -> None:
        """Returns the number of units"""

        if domain in (None, dict()):
            self.number_units["__none__"] = len(samp_weight)
        elif domain is not None:
            keys, values = np.unique(domain, return_counts=True)
            self.number_units = dict(zip(keys, values))

    @staticmethod
    def _deff_wgt(samp_weight: np.ndarray) -> float:
        """compute the design effect due to unequal weights -
        Page 71 of Valliant and Dever (2018) """

        mean_w = np.mean(samp_weight)
        relvar_w = np.power(samp_weight - mean_w, 2) / mean_w ** 2
        deff_w = 1 + np.mean(relvar_w)

        return deff_w

    def deff_weight(self, samp_weight: np.ndarray, domain: np.ndarray = None) -> Dict:
        """compute the design effect due to unequal weights across estimation scale_d.  

        Args:
            samp_weight (array) : Array of the pre-adjsutment sample 
            weight. This vector should contains numeric values.   

            domain (array) : Array indicating the scale_d of interest 
            for the calculation of the design effect due to weighting.

        
        Returns:
            A dictionary: the mapping of the scale_d with associated 
            design effect due to weighting.
        """

        deff_w = {}
        if domain is None:
            deff_w["__none__"] = self._deff_wgt(samp_weight)
        else:
            for d in np.unique(domain):
                deff_w[d] = self._deff_wgt(samp_weight[domain == d])
        self.deff_wgt = deff_w

        return deff_w

    @staticmethod
    def _norm_adjustment(
        samp_weight: np.ndarray, control: Union[float, Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:

        sum_weights = np.sum(samp_weight)
        norm_factor = control / sum_weights
        norm_weight = samp_weight * norm_factor

        return norm_weight, norm_factor

    @staticmethod
    def _response(resp_status: np.ndarray, resp_dict: np.ndarray) -> np.ndarray:

        resp_status = formats.numpy_array(resp_status)
        checks.check_response_status(resp_status, resp_dict)

        if not np.isin(resp_status, ("in", "rr", "nr", "uk")).any():
            resp_code = np.repeat("  ", resp_status.size).astype(str)
            resp_code[resp_status == resp_dict["in"]] = "in"
            resp_code[resp_status == resp_dict["rr"]] = "rr"
            resp_code[resp_status == resp_dict["nr"]] = "nr"
            resp_code[resp_status == resp_dict["uk"]] = "uk"
        else:
            resp_code = resp_status

        return resp_code

    @staticmethod
    def _adjust_factor(
        samp_weight: np.ndarray, resp_code: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        in_sample = resp_code == "in"  # ineligible
        rr_sample = resp_code == "rr"  # respondent
        nr_sample = resp_code == "nr"  # nonrespondent
        uk_sample = resp_code == "uk"  # unknown

        in_weights_sum = np.sum(samp_weight[in_sample])
        rr_weights_sum = np.sum(samp_weight[rr_sample])
        nr_weights_sum = np.sum(samp_weight[nr_sample])
        uk_weights_sum = np.sum(samp_weight[uk_sample])

        adjust_rr = (rr_weights_sum + nr_weights_sum + uk_weights_sum) / rr_weights_sum

        adjust_factor = np.ones(samp_weight.size)  # ineligibles will get 1 by default
        adjust_factor[rr_sample] = adjust_rr
        adjust_factor[nr_sample] = 0
        adjust_factor[uk_sample] = 0

        return adjust_factor, adjust_rr

    def adjust(
        self,
        samp_weight: np.ndarray,
        adjust_class: np.ndarray,
        resp_status: np.ndarray,
        resp_dict: Dict = None,
    ) -> np.ndarray:
        """
        adjust sample weight to account for non-response. 

        Args:
            samp_weight (array) : Array of the pre-adjsutment sample 
            weight. This vector should contains numeric values.   

            adjust_class (array) : Array indicating the adjustment 
            class for each sample unit. The sample weight adjustments 
            will be performed within the classes defined by this 
            parameter.

            resp_status (array) : Array indicating the eligibility 
            and response status of the sample unit. Values of 
            resp_status should inform on ineligible (in), 
            respondent (rr), nonrespondent (nr), 
            not known / unknown (uk). If the values of the paramter 
            are not in ("in", "rr", "nr", "uk") then the resp_dict 
            is required.

            resp_dict (dictionary) : Dictionnary providing the 
            mapping between the values of resp_status and the 
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

        resp_code = self._response(resp_status, resp_dict)
        samp_weight = formats.numpy_array(samp_weight)
        if adjust_class is not None:
            adjust_class = formats.non_missing_array(adjust_class)
        adjusted_weight = np.ones(samp_weight.size) * np.nan

        if adjust_class is None or adjust_class.size <= 1:
            adjust_factor, self.adjust_factor["__none__"] = self._adjust_factor(
                samp_weight, resp_code
            )
            adjusted_weight = adjust_factor * samp_weight
        else:
            for c in np.unique(adjust_class):
                samp_weight_c = samp_weight[adjust_class == c]
                resp_code_c = resp_code[adjust_class == c]
                adjust_factor_c, self.adjust_factor[c] = self._adjust_factor(
                    samp_weight_c, resp_code_c
                )
                adjusted_weight[adjust_class == c] = adjust_factor_c * samp_weight_c
        self.deff_wgt = self.deff_weight(adjusted_weight)

        return adjusted_weight

    @staticmethod
    def _core_matrix(
        samp_weight: np.ndarray,
        x: np.ndarray,
        x_weighted_total: np.ndarray,
        control: np.ndarray,
        scale: np.ndarray,
    ):

        v_inv_d = np.diag(samp_weight / scale)
        core_matrix = np.matmul(np.matmul(np.transpose(x), v_inv_d), x)

        core_matrix_inv = np.linalg.inv(core_matrix)
        core_factor = np.matmul(np.transpose(control - x_weighted_total), core_matrix_inv)
        return core_factor

    @staticmethod
    def _core_vector(x_i, core_factor):

        return np.dot(core_factor, x_i)

    def normalize(
        self,
        samp_weight: np.ndarray,
        control: Union[float, Dict[float, float]] = None,
        domain: np.ndarray = None,
    ) -> np.ndarray:
        """
        normalize the sample weights to sum to a known constants or 
        levels. 

        Args:
            samp_weight (array) : Array of the pre-adjsutment sample 
            weight. This vector should contains numeric values.   

            control (int, float, dictionary) : A number or array of 
            the level to calibrate the sum of the weights. Default is 
            number of units by domain key or overall if 
            domain is None. 

            domain (array) : Array indicating the normalization 
            class for each sample unit.
        
        Returns:
            An arrays: the normalized sample weight.
        """

        samp_weight = formats.numpy_array(samp_weight)
        norm_weight = samp_weight.copy()

        if domain is not None:
            domain = formats.numpy_array(domain)
            keys = np.unique(domain)
            levels: np.ndarray = np.zeros(keys.size) * np.nan
            for k, key in enumerate(keys):
                weight_k = samp_weight[domain == key]
                if control is None:
                    levels[k] = np.sum(domain == key)
                elif control is not None and not isinstance(control, (float, int)):
                    levels[k] = control[key]
                elif isinstance(control, (float, int)):
                    levels[k] = control
                (norm_weight[domain == key], self.norm_factor[key]) = self._norm_adjustment(
                    weight_k, levels[k]
                )
                self.control[key] = levels[k]
        else:
            if control is not None:
                level = control
            else:
                level = np.sum(samp_weight.size)
            norm_weight, self.norm_factor["__none__"] = self._norm_adjustment(samp_weight, level)
            self.control["__none__"] = level

        return norm_weight

    def poststratify(
        self,
        samp_weight: np.ndarray,
        control: Union[float, Dict[float, float]] = None,
        factor: Union[float, Dict[float, float]] = None,
        domain: np.ndarray = None,
    ) -> np.ndarray:

        if control is None and factor is None:
            raise AssertionError("control or factor must be specified.")

        if isinstance(control, dict):
            if (np.unique(domain) != np.unique(list(control.keys()))).any():
                raise ValueError("control dictionary keys do not much domain values.")

        if control is None and domain is not None:
            if (np.unique(domain) != np.unique(list(factor.keys()))).any():
                raise ValueError("factor dictionary keys do not much domain values.")

            sum_weight = np.sum(samp_weight)
            if isinstance(factor, dict):
                control = {}
                for d in np.unique(domain):
                    control[d] = sum_weight * factor[d]
            elif isinstance(factor, float):
                control = sum_weight * factor

        return self.normalize(samp_weight, control, domain)

    def _raked_wgt(self, samp_weight, X, control, domain):
        pass

    def rake(
        self,
        samp_weight: np.ndarray,
        x: np.ndarray,
        control: Union[float, Dict[float, float]],
        scale=1,
        domain=None,
    ) -> np.ndarray:

        pass

    @staticmethod
    def _calib_covariates(
        data: pd.DataFrame, x_cat: List[str] = None, x_cont: List[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:

        if not isinstance(data, pd.DataFrame) or data is None:
            raise ValueError("data must be a pandas dataframe.")
        if x_cat is None and x_cont is None:
            raise AssertionError("x_cat and/or x_cont must be specified.")

        if x_cat is not None:
            x_concat = formats.dataframe_to_array(data[x_cat])
            x_dummies = pd.get_dummies(x_concat)
            x_dict = formats.array_to_dict(x_concat)
            # x_dummies.insert(0, "intercept", 1)
        if x_cont is None:
            x_array = x_dummies.astype("int")
        else:
            x_array = pd.concat([x_dummies, data[x_cont]], axis=1).astype("int")
            x_cont_dict = {}
            nb_obs = data[x_cont].shape[0]
            for var in x_cont:
                x_cont_dict[var] = nb_obs
                x_dict.update(x_cont_dict)

        return x_array, x_dict

    def calib_covariates(
        self,
        data: pd.DataFrame,
        x_cat: List[str] = None,
        x_cont: List[str] = None,
        domain: List[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:

        if domain is None:
            x_array, x_dict = self._calib_covariates(data, x_cat, x_cont)
        else:
            x_array = pd.DataFrame()
            x_dict = {}
            for k, d in enumerate(np.unique(data[domain])):
                x_array_d, x_dict_d = self._calib_covariates(
                    data[data[domain] == d], x_cat, x_cont
                )
                x_array = x_array.append(x_array_d)
                x_dict[d] = x_dict_d

        return x_array.to_numpy(), x_dict

    def _calib_wgt(
        self, samp_weight: np.ndarray, x: np.ndarray, core_factor: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:

        adjust_factor = np.apply_along_axis(
            self._core_vector, axis=1, arr=x, core_factor=core_factor
        )
        adjusted_factor = 1 + adjust_factor / scale

        return adjusted_factor

    def calibrate(
        self,
        samp_weight: np.ndarray,
        x: np.ndarray,
        control: Dict[Any, Any] = None,
        domain: np.ndarray = None,
        scale: Union[np.ndarray, float] = 1,
        bounded: bool = False,
        modified: bool = False,
    ) -> np.ndarray:

        if not isinstance(samp_weight, np.ndarray):
            samp_weight = formats.numpy_array(samp_weight)
        if not isinstance(x, np.ndarray):
            x = formats.numpy_array(x)
        if isinstance(scale, (float, int)):
            scale = scale * np.ones(np.size(samp_weight))

        x_weighted_total = np.sum(np.transpose(x) * samp_weight[:], axis=1)
        if domain is None:
            core_factor = self._core_matrix(
                samp_weight=samp_weight,
                x=x,
                x_weighted_total=x_weighted_total,
                control=np.array(list(control.values())),
                scale=scale,
            )
            adjust_factor = self._calib_wgt(samp_weight, x, core_factor, scale)
        else:
            adjust_factor = []
            for d in np.unique(domain):
                x_d = x[domain == d]
                samp_weight_d = samp_weight[domain == d]
                x_weighted_total_d = np.sum(np.transpose(x_d) * samp_weight_d[:], axis=1)
                control_d = control[d]
                scale_d = scale[domain == d]
                if modified:
                    core_factor = self._core_matrix(
                        samp_weight=samp_weight_d,
                        x=x[domain == d],
                        x_weighted_total=x_weighted_total_d,
                        control=np.array(list(control_d.values())),
                        scale=scale_d,
                    )
                else:
                    core_factor = self._core_matrix(
                        samp_weight=samp_weight,
                        x=x,
                        x_weighted_total=x_weighted_total_d,
                        control=np.array(list(control_d.values())),
                        scale=scale,
                    )
                adjust_factor_d = self._calib_wgt(samp_weight_d, x_d, core_factor, scale_d)
                adjust_factor = np.append(adjust_factor, adjust_factor_d)

        return samp_weight * adjust_factor

    def trim(
        self,
        samp_weight: np.ndarray,
        method: str,
        threshold: Union[float, Dict[float, float]],
        domain: np.ndarray = None,
    ) -> np.ndarray:
        """
        trim sample weight to reduce the influence of extreme weights. 

        Args:
            samp_weight (array) : Array of the pre-adjsutment sample 
            weight. This vector should contains numeric values.   

            method (string) : Name of the trimming method. 
            Possible values are: "threshold", "interquartile", and "?"

            threshold (int, float, dictionary) : A number defining 
            the threshold or a dictionnary mapping triming classes 
            to trimming thresholds. Depending on the trimming method, 
            this parameter may provide the 

            domain (array) : Array indicating the trimming class 
            for each sample unit.

        
        Returns:
            An array: the trimmed sample weight.
        """

        pass
