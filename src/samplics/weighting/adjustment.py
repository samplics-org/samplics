"""
Author: Mamadou S Diallo <msdiallo@QuantifyAfrica.org>


License: MIT
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


class SampleWeight:
    """Sample weight 
    """

    def __init__(self) -> None:

        self.number_scale_d = 1
        self.number_units: Dict[StringNumber, int] = {}
        self.deff_wgt: Dict[StringNumber, Number] = {}
        self.norm_factor: Dict[StringNumber, Number] = {}
        self.control: Dict[StringNumber, Number] = {}
        self.adjust_factor: Dict[StringNumber, Number] = {}
        self.trim_method = ""
        self.calib_factor = ""

    def __repr__(self) -> None:
        pass

    def __str__(self) -> None:
        pass

    def _number_units(self, domain: np.ndarray, samp_weight: np.ndarray) -> None:
        """Returns the number of units"""

        if domain in (None, dict()):
            self.number_units["__none__"] = len(samp_weight)
        elif domain is not None:
            keys, values = np.unique(domain, return_counts=True)
            self.number_units = dict(zip(keys, values))

    @staticmethod
    def _deff_wgt(samp_weight: np.ndarray) -> Number:
        """compute the design effect due to unequal weights -
        Page 71 of Valliant and Dever (2018) """

        mean_w = np.mean(samp_weight)
        relvar_w = np.power(samp_weight - mean_w, 2) / mean_w ** 2
        deff_w = 1 + np.mean(relvar_w)

        return deff_w

    def deff_weight(
        self, samp_weight: np.ndarray, domain: Optional[np.ndarray] = None
    ) -> Dict[StringNumber, Number]:
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

        deff_w: Dict[StringNumber, Number] = {}
        if domain is None:
            deff_w["__none__"] = self._deff_wgt(samp_weight)
        else:
            for d in np.unique(domain):
                deff_w[d] = self._deff_wgt(samp_weight[domain == d])
        self.deff_wgt = deff_w

        return deff_w

    @staticmethod
    def _norm_adjustment(
        samp_weight: np.ndarray, control: Union[Dict[StringNumber, Number], Number]
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
        samp_weight: np.ndarray, resp_code: np.ndarray, unknown_to_inelig: bool
    ) -> Tuple[np.ndarray, np.ndarray]:

        in_sample = resp_code == "in"  # ineligible
        rr_sample = resp_code == "rr"  # respondent
        nr_sample = resp_code == "nr"  # nonrespondent
        uk_sample = resp_code == "uk"  # unknown

        in_weights_sum = np.sum(samp_weight[in_sample])
        rr_weights_sum = np.sum(samp_weight[rr_sample])
        nr_weights_sum = np.sum(samp_weight[nr_sample])
        uk_weights_sum = np.sum(samp_weight[uk_sample])

        if unknown_to_inelig:
            adjust_uk = (in_weights_sum + rr_weights_sum + nr_weights_sum + uk_weights_sum) / (
                in_weights_sum + rr_weights_sum + nr_weights_sum
            )
            adjust_rr = (rr_weights_sum + nr_weights_sum) / rr_weights_sum
        else:
            adjust_uk = 1
            adjust_rr = (rr_weights_sum + nr_weights_sum + uk_weights_sum) / rr_weights_sum

        adjust_factor = np.zeros(samp_weight.size)  # unknown and nonresponse will get 1 by default
        adjust_factor[rr_sample] = adjust_rr * adjust_uk
        adjust_factor[in_sample] = adjust_uk

        return adjust_factor, adjust_rr

    def adjust(
        self,
        samp_weight: np.ndarray,
        adjust_class: np.ndarray,
        resp_status: np.ndarray,
        resp_dict: Union[Dict[str, StringNumber], None] = None,
        unknown_to_inelig: bool = True,
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
        adjusted_weight = np.ones(samp_weight.size) * np.nan

        if adjust_class is None:
            adjust_factor, self.adjust_factor["__none__"] = self._adjust_factor(
                samp_weight, resp_code, unknown_to_inelig
            )
            adjusted_weight = adjust_factor * samp_weight
        else:
            if isinstance(adjust_class, list):
                adjust_class = pd.DataFrame(np.column_stack(adjust_class))
            elif isinstance(adjust_class, np.ndarray):
                adjust_class = pd.DataFrame(adjust_class)
            elif not isinstance(adjust_class, (pd.Series, pd.DataFrame)):
                raise AssertionError(
                    "adjsut_class must be an numpy ndarray, a list of numpy ndarray or a pandas dataframe."
                )

            adjust_array = formats.dataframe_to_array(adjust_class)

            for c in np.unique(adjust_array):
                samp_weight_c = samp_weight[adjust_array == c]
                resp_code_c = resp_code[adjust_array == c]
                adjust_factor_c, self.adjust_factor[c] = self._adjust_factor(
                    samp_weight_c, resp_code_c, unknown_to_inelig
                )
                adjusted_weight[adjust_array == c] = adjust_factor_c * samp_weight_c
        self.deff_wgt = self.deff_weight(adjusted_weight)

        return adjusted_weight

    @staticmethod
    def _core_matrix(
        samp_weight: np.ndarray,
        x: np.ndarray,
        x_weighted_total: np.ndarray,
        x_control: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:

        v_inv_d = np.diag(samp_weight / scale)
        core_matrix = np.dot(np.matmul(np.transpose(x), v_inv_d), x)
        if x.shape == (x.size,):
            core_factor = (x_control - x_weighted_total) / core_matrix
        else:
            core_factor = np.matmul(
                np.transpose(x_control - x_weighted_total), np.linalg.inv(core_matrix)
            )

        return core_factor

    def normalize(
        self,
        samp_weight: Array,
        control: Union[Dict[StringNumber, Number], Number, None] = None,
        domain: Optional[Array] = None,
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
                elif control is not None and isinstance(control, Dict):
                    levels[k] = control[key]
                elif isinstance(control, (float, int)):
                    levels[k] = control

                (norm_weight[domain == key], self.norm_factor[key]) = self._norm_adjustment(
                    weight_k, levels[k]
                )
                self.control[key] = levels[k]
        else:
            if control is None:
                control = {"__none__": np.sum(samp_weight.size).astype("int")}
            elif isinstance(control, (int, float)):
                control = {"__none__": control}

            norm_weight, self.norm_factor["__none__"] = self._norm_adjustment(
                samp_weight, control["__none__"]
            )
            self.control["__none__"] = control["__none__"]

        return norm_weight

    def poststratify(
        self,
        samp_weight: Array,
        control: Union[Dict[StringNumber, Number], None] = None,
        factor: Union[Dict[StringNumber, Number], None] = None,
        domain: Optional[Array] = None,
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

    def _raked_wgt(
        self,
        samp_weight: np.ndarray,
        X: np.ndarray,
        control: Union[Dict[StringNumber, Number], None],
        domain: Optional[Array] = None,
    ) -> None:
        pass

    def rake(
        self,
        samp_weight: Array,
        x: Union[np.ndarray, pd.DataFrame],
        control: Union[Dict[StringNumber, Number], None] = None,
        scale: Union[np.ndarray, Number] = 1,
        domain: Optional[Array] = None,
    ) -> np.ndarray:

        pass

    @staticmethod
    def _calib_covariates(
        data: pd.DataFrame, x_cat: Optional[List[str]] = None, x_cont: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[StringNumber, Number]]:

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

        return x_array.to_numpy(), x_dict

    def calib_covariates(
        self,
        data: pd.DataFrame,
        x_cat: Optional[List[str]] = None,
        x_cont: Optional[List[str]] = None,
        domain: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Union[DictStrNum, Dict[StringNumber, DictStrNum]]]:

        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise AssertionError("data must be a pandas dataframe.")

        if isinstance(data[x_cat], pd.Series):
            nb_cols = (data[x_cat].drop_duplicates()).shape[0] + 1
        elif x_cont is None:
            nb_cols = (data[x_cat].drop_duplicates()).shape[0]
        else:
            nb_cols = (data[x_cat].drop_duplicates()).shape[0] + len(x_cont)

        x_dict: Dict[StringNumber, Dict[StringNumber, Number]] = {}
        if domain is None:
            x_array, x_dict["__none__"] = self._calib_covariates(data, x_cat, x_cont)
            for key in x_dict["__none__"]:
                x_dict["__none__"][key] = np.nan
        else:
            x_array = np.zeros((data.shape[0], nb_cols))
            for d in np.unique(data[domain].values):
                x_array[data[domain] == d, :], x_dict[d] = self._calib_covariates(
                    data[data[domain] == d], x_cat, x_cont
                )
                for key in x_dict[d]:
                    x_dict[d][key] = np.nan

        if domain is None:
            return x_array, x_dict["__none__"]
        else:
            return x_array, x_dict

    def _calib_wgt(self, x: np.ndarray, core_factor: np.ndarray) -> np.ndarray:
        def _core_vector(x_i: np.ndarray, core_factor: np.ndarray) -> np.ndarray:
            return np.dot(core_factor, x_i)

        if x.shape == (x.size,):
            adjust_factor = _core_vector(x, float(core_factor))
        else:
            adjust_factor = np.apply_along_axis(
                _core_vector, axis=1, arr=x, core_factor=core_factor
            )

        return adjust_factor

    def calibrate(
        self,
        samp_weight: np.ndarray,
        aux_vars: np.ndarray,
        control: Union[Dict[StringNumber, Union[DictStrNum, Number]], None] = None,
        domain: Optional[np.ndarray] = None,
        scale: Union[np.ndarray, Number] = 1,
        bounded: bool = False,
        additive: bool = False,
    ) -> np.ndarray:

        samp_size = samp_weight.size

        if not isinstance(samp_weight, np.ndarray):
            samp_weight = formats.numpy_array(samp_weight)
        if not isinstance(aux_vars, np.ndarray):
            aux_vars = formats.numpy_array(aux_vars)
        if domain is not None and not isinstance(domain, np.ndarray):
            domain = formats.numpy_array(domain)
        if isinstance(scale, (float, int)):
            scale = np.repeat(scale, samp_size)

        if aux_vars.shape == (samp_size,):  # one dimentional array
            x_w = aux_vars * samp_weight
            one_dimension = True
        else:
            x_w = np.transpose(aux_vars) * samp_weight
            one_dimension = False

        if domain is None:
            if one_dimension:
                x_w_total = np.sum(x_w)
            else:
                x_w_total = np.sum(x_w, axis=1)
            core_factor = self._core_matrix(
                samp_weight=samp_weight,
                x=aux_vars,
                x_weighted_total=x_w_total,
                x_control=np.array(list(control.values())),
                scale=scale,
            )
            adjust_factor = 1 + self._calib_wgt(aux_vars, core_factor) / scale
        else:
            domains = np.unique(domain)
            if additive:
                adjust_factor = np.ones((samp_size, domains.size)) * np.nan
            else:
                adjust_factor = np.ones(samp_size) * np.nan

            for k, d in enumerate(domains):
                if one_dimension:
                    x_w_total = np.sum(x_w)
                else:
                    x_w_total = np.sum(x_w, axis=1)

                x_d = aux_vars[domain == d]
                samp_weight_d = samp_weight[domain == d]
                if one_dimension:  # one dimentional array
                    x_w_total_d = np.sum(x_w[domain == d])
                else:
                    x_w_total_d = np.sum(np.transpose(x_w)[domain == d], axis=0)

                control_d = control.get(d)
                if isinstance(control_d, (int, float)):
                    control_d_values = [control_d]
                elif isinstance(control_d, Dict):
                    control_d_values = list(control_d.values())

                scale_d = scale[domain == d]
                if additive:
                    core_factor_d = self._core_matrix(
                        samp_weight=samp_weight,
                        x=aux_vars,
                        x_weighted_total=x_w_total_d,
                        x_control=np.array(control_d_values),
                        scale=scale,
                    )
                    adjust_factor[:, k] = (domain == d) + self._calib_wgt(
                        aux_vars, core_factor_d
                    ) / scale
                else:
                    core_factor_d = self._core_matrix(
                        samp_weight=samp_weight_d,
                        x=aux_vars[domain == d],
                        x_weighted_total=x_w_total_d,
                        x_control=np.array(control_d_values),
                        scale=scale_d,
                    )
                    adjust_factor[domain == d] = 1 + self._calib_wgt(x_d, core_factor_d) / scale_d
                # adjust_factor = np.append(adjust_factor, adjust_factor_d)

        if additive:
            calib_weight = np.transpose(np.transpose(adjust_factor) * samp_weight)
        else:
            calib_weight = samp_weight * adjust_factor

        return calib_weight

    def trim(
        self,
        samp_weight: Array,
        method: str,
        threshold: Union[Dict[StringNumber, Union[DictStrNum, Number]], None],
        domain: Optional[Array] = None,
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
