"""Sample weighting module

*SampleWeight* is the main class in this module which implements weight adjustments to account for 
nonresponse, calibrate to auxiliary information, normalize weights, and trim extreme weights. Valliant, R. and Dever, J. A. (2018) [#vd2018]_ provides a step-by-step guide on calculating 
sample weights. 

.. [#vd2018] Valliant, R. and Dever, J. A. (2018), *Survey Weights: A Step-by-Step Guide to       
   Calculation*, Stata Press.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from samplics.utils import checks, formats
from samplics.utils.types import Array, DictStrFloat, DictStrInt, DictStrNum, Number, StringNumber


class SampleWeight:
    """*SampleWeight* implements several adjustments to sample weights. The class does not computes design sample weights. It is expected at this point some initial weights are
    available e.g. design sample weights or some other sample weights. Using this module,
    the user will be able to adjust sample weights to account for nonresponse, normalize
    sample weights so that they sum to some control value(s), poststratify, and calibrate
    based on auxiliary information.

    Attributes
        | adjust_method (str): adjustment method. Possible values are nonresponse,
        |   normalization, poststratification, calibration.
        | number_units (dict): number of units per domain.
        | deff_wgt (dict): design effect due to unequal weights per domain.
        | adjust_factor (dict): normalizing adjustment factor per domain.
        | control (dict): control values per domain.

    Methods
        | deff_weight(): computes the design effect due to weighting.
        | adjust(): adjust the sample weights to account for nonresponse.
        | normalize(): normalize the sample weights to ensure they sum to a control value.
        | poststratify(): poststratify the sample weights.
        | calib_covariables(): covert a dataframe to a tuple of an array and a dictionary.
        | The array corresponds to the calibration domains. The dictionary maps the array
        | elements with their corresponding control values.
        | calibrate(): calibrate the sample weights.

    TODO: trim(), rake()
    """

    def __init__(self) -> None:

        self.adjust_method: str = ""
        self.number_units: Union[DictStrInt, int] = {}
        self.deff_wgt: Union[DictStrNum, Number] = {}
        self.adjust_factor: Union[DictStrNum, Number] = {}
        self.control: Union[DictStrNum, Number] = {}

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass

    def _number_units(self, domain: Optional[np.ndarray], samp_weight: np.ndarray) -> None:
        """Returns the number of units"""

        if domain is None:
            self.number_units = len(samp_weight)
        elif domain is not None:
            keys, values = np.unique(domain, return_counts=True)
            self.number_units = dict(zip(keys, values))

    @staticmethod
    def _deff_wgt(samp_weight: np.ndarray) -> Number:
        """compute the design effect due to unequal weights -
        Page 71 of Valliant and Dever (2018)"""

        mean_w = np.mean(samp_weight)
        relvar_w = np.power(samp_weight - mean_w, 2) / mean_w**2

        return float(1 + np.mean(relvar_w))

    def deff_weight(
        self, samp_weight: Array, domain: Optional[np.ndarray] = None
    ) -> Union[DictStrNum, Number]:
        """Computes the design effect due to unequal weights.

        Args:
            samp_weight (Array):  array of the pre-adjustment sample weight. This vector
                should contains numeric values.
            domain (Optional[np.ndarray], optional): array indicating the normalization class
                for each sample unit. Defaults to None. Defaults to None.

        Returns:
            DictStrNum: dictionnary pairing the domains to the design effects due
                unequal weights.
        """

        samp_weight = formats.numpy_array(samp_weight)

        if domain is None:
            self.deff_wgt = self._deff_wgt(samp_weight)
            return self.deff_wgt
        else:
            self.deff_wgt = {}
            for d in np.unique(domain):
                self.deff_wgt[d] = self._deff_wgt(samp_weight[domain == d])
            return self.deff_wgt

    @staticmethod
    def _norm_adjustment(
        samp_weight: np.ndarray,
        control: Number,
    ) -> tuple[np.ndarray, Number]:

        sum_weights = np.sum(samp_weight)
        adjust_factor = float(control / sum_weights)

        return np.asarray(samp_weight * adjust_factor), adjust_factor

    @staticmethod
    def _response(
        resp_status: np.ndarray, resp_dict: Optional[dict[str, StringNumber]]
    ) -> np.ndarray:

        resp_status = formats.numpy_array(resp_status)
        checks.assert_response_status(resp_status, resp_dict)

        if not np.isin(resp_status, ("in", "rr", "nr", "uk")).any() and resp_dict is not None:
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
    ) -> tuple[np.ndarray, Number]:

        in_sample = resp_code == "in"  # ineligible
        rr_sample = resp_code == "rr"  # respondent
        nr_sample = resp_code == "nr"  # nonrespondent
        uk_sample = resp_code == "uk"  # unknown

        in_weights_sum = float(np.sum(samp_weight[in_sample]))
        rr_weights_sum = float(np.sum(samp_weight[rr_sample]))
        nr_weights_sum = float(np.sum(samp_weight[nr_sample]))
        uk_weights_sum = float(np.sum(samp_weight[uk_sample]))

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
        samp_weight: Array,
        adjust_class: Array,
        resp_status: Array,
        resp_dict: Optional[Union[dict[str, StringNumber]]] = None,
        unknown_to_inelig: bool = True,
    ) -> np.ndarray:
        """adjusts sample weight to account for non-response.

        Args:
            samp_weight (np.ndarray): array of the pre-adjustment sample weight. This vector
                should contains numeric values.
            adjust_class (np.ndarray): array indicating the adjustment class for each sample unit.
                The sample weight adjustments will be performed within the classes defined by this
                parameter.
            resp_status (np.ndarray): array indicating the eligibility and response status of the
                sample unit. Values of resp_status should inform on ineligible (in), respondent (rr), nonrespondent (nr), not known / unknown (uk). If the values of the parameter are not in ("in", "rr", "nr", "uk") then the resp_dict is required.
            resp_dict (Union[dict[str, StringNumber], None], optional): dictionnary providing the
                mapping between the values of resp_status and the ["in", "rr", "nr", "uk"].
                For example, if the response status are: 0 for ineligible, 1 for respondent,
                2 for nonrespondent, and 9 for unknown. Then the dictionary will be {"in": 0, "rr": 1, "nr": 2, "uk": 9}. If the response status variable has only values in ("in", "rr", "nr", "uk") then the dictionary is not needed. Optional parameter. Defaults to None.
            unknown_to_inelig (bool, optional): [description]. Defaults to True.

        Raises:
            AssertionError: raises an assertion error if adjust_class is not a list, numpy array,
            or pandas dataframe/series.

        Returns:
            np.ndarray: array of the adjusted sample weights.
        """

        resp_code = self._response(formats.numpy_array(resp_status), resp_dict)
        samp_weight = formats.numpy_array(samp_weight)
        adjusted_weight = np.ones(samp_weight.size) * np.nan

        if adjust_class is None:
            (
                adjust_factor,
                self.adjust_factor,
            ) = self._adjust_factor(samp_weight, resp_code, unknown_to_inelig)
            adjusted_weight = adjust_factor * samp_weight
        else:
            if isinstance(adjust_class, list):
                adjust_class = pd.DataFrame(np.column_stack(adjust_class))
            elif isinstance(adjust_class, np.ndarray):
                adjust_class = pd.DataFrame(adjust_class)
            elif not isinstance(adjust_class, (pd.Series, pd.DataFrame)):
                raise AssertionError(
                    "adjust_class must be an numpy ndarray, a list of numpy ndarray or a pandas dataframe."
                )

            adjust_array = formats.dataframe_to_array(adjust_class)
            self.adjust_factor = {}
            for c in np.unique(adjust_array):
                samp_weight_c = samp_weight[adjust_array == c]
                resp_code_c = resp_code[adjust_array == c]
                adjust_factor_c, self.adjust_factor[c] = self._adjust_factor(
                    samp_weight_c, resp_code_c, unknown_to_inelig
                )
                adjusted_weight[adjust_array == c] = adjust_factor_c * samp_weight_c

        self.deff_wgt = self.deff_weight(adjusted_weight)
        self.adjust_method = "nonresponse"

        return np.asarray(adjusted_weight)

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
                np.transpose(x_control - x_weighted_total),
                np.linalg.inv(core_matrix),
            )

        return np.asarray(core_factor)

    def normalize(
        self,
        samp_weight: Array,
        control: Optional[Union[DictStrNum, Number]] = None,
        domain: Optional[Array] = None,
    ) -> np.ndarray:
        """normalizes the sample weights to sum to a known constants or levels.

        Args:
            samp_weight (array) : array of the pre-adjustment sample weight. This vector should
                contains numeric values.
            control (int, float, dictionary) : a number or array of the level to calibrate the
                sum of the weights. Default is number of units by domain key or overall if domain
                is None. Defaults to None.
            domain (Optional[Array], optional) : array indicating the normalization class for each
                sample unit. Defaults to None.

        Returns:
            An arrays: the normalized sample weight.
        """

        samp_weight = formats.numpy_array(samp_weight)
        norm_weight = samp_weight.copy()

        if domain is not None:
            domain = formats.numpy_array(domain)
            keys = np.unique(domain)
            levels: np.ndarray = np.zeros(keys.size) * np.nan
            self.adjust_factor = {}
            self.control = {}
            for k, key in enumerate(keys):
                weight_k = samp_weight[domain == key]
                if control is None:
                    levels[k] = np.sum(domain == key)
                elif control is not None and isinstance(control, dict):
                    levels[k] = control[key]
                elif isinstance(control, (float, int)):
                    levels[k] = control

                (
                    norm_weight[domain == key],
                    self.adjust_factor[key],
                ) = self._norm_adjustment(weight_k, levels[k])
                self.control[key] = levels[k]
        else:
            if control is None:
                self.control = int(np.sum(samp_weight.size))
                norm_weight, self.adjust_factor = self._norm_adjustment(samp_weight, self.control)
            elif isinstance(control, (int, float)):
                norm_weight, self.adjust_factor = self._norm_adjustment(samp_weight, control)
                self.control = control

        self.adjust_method = "normalization"

        return norm_weight

    def poststratify(
        self,
        samp_weight: Array,
        control: Optional[Union[DictStrNum, Number]] = None,
        factor: Optional[Union[DictStrNum, Number]] = None,
        domain: Optional[Array] = None,
    ) -> np.ndarray:
        """[summary]

        Args:
            samp_weight (Array): [description]
            control (Union[DictStrNum, Number, None], optional): a number or
                array of the level to calibrate the sum of the weights. Defaults to None.
            factor (Union[DictStrNum, Number, None], optional): adjustment factor.
                Defaults to None.
            domain (Optional[Array], optional): array indicating the normalization class for each
                sample unit. Defaults to None.

        Raises:
            AssertionError: raises an assertion error if both control and factor are not provided.
            ValueError: raises an error is control dictionary keys do not match domain's values.
            ValueError: raises an error is factor dictionary keys do not match domain's values.

        Returns:
            np.ndarray:  array of poststratified sample weights.
        """

        if control is None and factor is None:
            raise AssertionError("control or factor must be specified.")

        if isinstance(control, dict):
            if (np.unique(domain) != np.unique(list(control.keys()))).any():
                raise ValueError("control dictionary keys do not match domain values.")

        if control is None and domain is not None:
            if (
                isinstance(factor, dict)
                and (np.unique(domain) != np.unique(list(factor.keys()))).any()
            ):
                raise ValueError("factor dictionary keys do not match domain values.")

            sum_weight = float(np.sum(samp_weight))
            if isinstance(factor, dict):
                control = {}
                for d in np.unique(domain):
                    control[d] = sum_weight * factor[d]
            elif isinstance(factor, (int, float)):
                control = sum_weight * factor

        ps_weight = self.normalize(samp_weight, control, domain)
        self.adjust_method = "poststratification"

        return ps_weight

    def _raked_wgt(
        self,
        samp_weight: np.ndarray,
        X: np.ndarray,
        control: Union[DictStrNum, None],
        domain: Optional[Array] = None,
    ) -> None:
        pass

    def rake(
        self,
        samp_weight: Array,
        x: Union[np.ndarray, pd.DataFrame],
        control: Union[DictStrNum, None] = None,
        scale: Union[np.ndarray, Number] = 1,
        domain: Optional[Array] = None,
    ) -> np.ndarray:

        pass

    @staticmethod
    def _calib_covariates(
        data: pd.DataFrame,
        x_cat: Optional[list[str]] = None,
        x_cont: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, DictStrNum]:

        if not isinstance(data, pd.DataFrame) or data is None:
            raise ValueError("data must be a pandas dataframe.")

        if x_cat is None and x_cont is None:
            raise AssertionError("x_cat and/or x_cont must be specified.")
        else:
            if x_cat is not None:
                x_concat = formats.dataframe_to_array(data[x_cat])
                x_dummies = pd.get_dummies(x_concat)
                x_dict = formats.array_to_dict(x_concat)
                # x_dummies.insert(0, "intercept", 1)
            if x_cont is None and x_dummies is not None:
                x_array = x_dummies.astype("int")
            elif x_cont is not None and x_dict is not None:
                x_array = pd.concat([x_dummies, data[x_cont]], axis=1).astype("int")
                x_cont_dict: DictStrNum = {}
                nb_obs = data[x_cont].shape[0]
                for var in x_cont:
                    x_cont_dict[var] = nb_obs
                    x_dict.update(x_cont_dict)
            else:
                raise AssertionError

        return np.asarray(x_array.to_numpy()), x_dict

    def calib_covariates(
        self,
        data: pd.DataFrame,
        x_cat: Optional[list[str]] = None,
        x_cont: Optional[list[str]] = None,
        domain: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, Union[DictStrNum, dict[StringNumber, DictStrNum]]]:
        """A utility function that creates an array of the calibration groups/domains and
        a dictionary pairing the domains with the control values.

        Args:
            data (pd.DataFrame): input pandas dataframe with the calibration's control data.
            x_cat (Optional[list[str]], optional): list of the names of the categorical control
                variables. Defaults to None.
            x_cont (Optional[list[str]], optional): list of the names of the continuous control
                variables. Defaults to None.
            domain (Optional[list[str]], optional): list of the names of the variables defining
                the normalization classes for each sample unit. Defaults to None.

        Raises:
            AssertionError: raises an assertion error if input data is not a pandas dataframe.

        Returns:
            tuple[np.ndarray, Union[DictStrNum, dict[StringNumber, DictStrNum]]]: a tuple of
            an array of the calibration domains and a dictionary pairing the domains with the
            control values.
        """

        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise AssertionError("data must be a pandas dataframe.")

        if isinstance(data[x_cat], pd.Series):
            nb_cols = (data[x_cat].drop_duplicates()).shape[0] + 1
        elif x_cont is None:
            nb_cols = (data[x_cat].drop_duplicates()).shape[0]
        else:
            nb_cols = (data[x_cat].drop_duplicates()).shape[0] + len(x_cont)

        x_dict: Union[DictStrNum, dict[StringNumber, DictStrNum]]
        if domain is None:
            x_array, x_dict = self._calib_covariates(data, x_cat, x_cont)
            for key in x_dict:
                x_dict[key] = np.nan
        else:
            x_dict2: dict[StringNumber, DictStrNum] = {}
            x_dict_d: DictStrNum
            x_array = np.zeros((data.shape[0], nb_cols))
            for d in np.unique(data[domain].values):
                (
                    x_array[data[domain] == d, :],
                    x_dict_d,
                ) = self._calib_covariates(data[data[domain] == d], x_cat, x_cont)
                for key in x_dict_d:
                    x_dict_d[key] = np.nan
                x_dict2[d] = x_dict_d
            x_dict = x_dict2

        if domain is None:
            return x_array, x_dict
        else:
            return x_array, x_dict

    def _calib_wgt(self, x: np.ndarray, core_factor: np.ndarray) -> np.ndarray:
        def _core_vector(x_i: np.ndarray, core_factor: np.ndarray) -> np.ndarray:
            return np.asarray(np.dot(core_factor, x_i))

        if x.shape == (x.size,):
            adjust_factor = _core_vector(x, core_factor)
        else:
            adjust_factor = np.apply_along_axis(
                _core_vector, axis=1, arr=x, core_factor=core_factor
            )

        return adjust_factor

    def calibrate(
        self,
        samp_weight: Array,
        aux_vars: Array,
        control: Union[dict[StringNumber, Union[DictStrNum, Number]]],
        domain: Optional[Array] = None,
        scale: Union[Array, Number] = 1,
        bounded: bool = False,
        additive: bool = False,
    ) -> np.ndarray:
        """Calibrates the sample weights.

        Args:
            samp_weight (Array): array of sample weights.
            aux_vars (Array): array of auxiliary variables.
            control (Union[dict[StringNumber, Union[DictStrNum, Number]], None], optional):
                provides the controls by domain if applicable. Defaults to None.
            domain (Optional[Array], optional): Array indicating the normalization class for each
                sample unit. Defaults to None.
            scale (Union[Array, Number], optional): [description]. Defaults to 1.
            bounded (bool, optional): [description]. Defaults to False.
            additive (bool, optional): [description]. Defaults to False.

        Returns:
            np.ndarray: an array of the calibrated sample weights.
        """

        samp_weight = formats.numpy_array(samp_weight)
        aux_vars = formats.numpy_array(aux_vars)
        samp_size = samp_weight.size
        if domain is not None:
            domain = formats.numpy_array(domain)
        if isinstance(scale, (float, int)):
            scale = np.repeat(scale, samp_size)
        else:
            scale = formats.numpy_array(scale)
        if aux_vars.shape == (samp_size,):
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
                if one_dimension:
                    x_w_total_d = np.sum(x_w[domain == d])
                else:
                    x_w_total_d = np.sum(np.transpose(x_w)[domain == d], axis=0)

                control_d = control.get(d)
                if isinstance(control_d, (int, float)):
                    control_d_values = [control_d]
                elif isinstance(control_d, dict):
                    control_d_values = list(control_d.values())
                else:
                    raise TypeError("Type of control not valid!")

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

        if additive:
            calib_weight = np.transpose(np.transpose(adjust_factor) * samp_weight)
        else:
            calib_weight = samp_weight * adjust_factor

        self.adjust_method = "calibration"

        return calib_weight

    def trim(
        self,
    ) -> np.ndarray:

        pass
