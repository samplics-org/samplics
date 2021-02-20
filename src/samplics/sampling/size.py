"""Sample size calculation module 

"""

from __future__ import annotations

from typing import Any, Union, Optional

import math

import numpy as np
import pandas as pd

from scipy.stats import norm as normal

from samplics.utils import formats
from samplics.utils.types import Array, Number, StringNumber, DictStrNum


class SampleSize:
    """*SampleSize* implements sample size calculation methods"""

    def __init__(
        self, parameter: str = "proportion", method: str = "wald", stratification: bool = False
    ) -> None:

        self.parameter = parameter.lower()
        self.method = method.lower()
        if self.method not in ("wald", "fleiss"):
            raise AssertionError("Sample size calculation method not valid.")
        self.stratification = stratification
        self.target: Union[DictStrNum, Number]
        self.samp_size: Union[DictStrNum, Number] = 0
        self.deff_c: Union[DictStrNum, Number] = 1.0
        self.deff_w: Union[DictStrNum, Number] = 1.0
        self.precision: Union[DictStrNum, Number] 
        self.resp_rate: Union[DictStrNum, Number] = 1.0

    def icc(self) -> Union[DictStrNum, Number]:
        pass

    def deff(
        self,
        cluster_size: Union[DictStrNum, Number],
        icc: Union[DictStrNum, Number],
    ) -> Union[DictStrNum, Number]:

        if isinstance(cluster_size, (int, float)) and isinstance(icc, (int, float)):
            return max(1 + (cluster_size - 1) * icc, 0)
        elif isinstance(cluster_size, dict) and isinstance(icc, dict):
            if cluster_size.keys() != icc.keys():
                raise AssertionError("Parameters do not have the same dictionary keys.")
            deff_c: DictStrNum = {}
            for s in cluster_size:
                deff_c[s] = max(1 + (cluster_size[s] - 1) * icc[s], 0)
            return deff_c
        else:
            raise ValueError("Combination of types not supported.")

    def _calculate_wald(
        self,
        target: Union[DictStrNum, Number],
        precision: Union[DictStrNum, Number],
        stratum: Optional[Array],
    ) -> Union[DictStrNum, Number]:

        z_value = normal().ppf(1 - self.alpha / 2)

        if (
            isinstance(target, dict)
            and isinstance(precision, dict)
            and isinstance(self.deff_c, dict)
            and stratum is not None
        ):
            samp_size: DictStrNum = {}
            for s in stratum:
                samp_size[s] = math.ceil(
                    self.deff_c[s] * z_value ** 2 * target[s] * (1 - target[s]) / precision[s] ** 2
                )
            return samp_size
        elif (
            isinstance(target, (int, float))
            and isinstance(precision, (int, float))
            and isinstance(self.deff_c, (int, float))
        ):
            return math.ceil(self.deff_c * z_value ** 2 * target * (1 - target) / precision ** 2)
        else:
            raise TypeError("target and precision must be numbers or dictionnaires!")

    def _calculate_fleiss(
        self,
        target: Union[DictStrNum, Number],
        precision: Union[DictStrNum, Number],
        stratum: Optional[Array],
    ) -> Union[DictStrNum, Number]:

        z_value = normal().ppf(1 - self.alpha / 2)

        def fleiss_factor(p: float, d: float) -> float:

            if 0 <= p < d or 1 - d < p <= 1:
                return 8 * d * (1 - 2 * d)
            elif d <= p < 0.3:
                return 4 * (p + d) * (1 - p - d)
            elif 0.7 < p <= 1 - d:
                return 4 * (p - d) * (1 - p + d)
            elif 0.3 <= p <= 0.7:
                return 1
            else:
                raise ValueError("Parameters p or d not valid.")

        if (
            self.stratification
            and isinstance(target, dict)
            and isinstance(precision, dict)
            and isinstance(self.deff_c, dict)
            and stratum is not None
        ):
            samp_size: DictStrNum = {}
            for s in stratum:
                fct = fleiss_factor(target[s], precision[s])
                samp_size[s] = math.ceil(
                    self.deff_c[s]
                    * (
                        fct * (z_value ** 2) / (4 * precision[s] ** 2)
                        + 1 / precision[s]
                        - 2 * z_value ** 2
                        + (z_value + 2) / fct
                    )
                )
            return samp_size
        elif (
            not self.stratification
            and isinstance(target, (int, float))
            and isinstance(precision, (int, float))
            and isinstance(self.deff_c, (int, float))
        ):
            fct = fleiss_factor(target, precision)
            return math.ceil(
                self.deff_c
                * (
                    fct * (z_value ** 2) / (4 * precision ** 2)
                    + 1 / precision
                    - 2 * z_value ** 2
                    + (z_value + 2) / fct
                )
            )
        else:
            raise TypeError("target and precision must be numbers or dictionnaires!")

    def calculate(
        self,
        target: Union[DictStrNum, Number],
        precision: Union[DictStrNum, Number],
        deff: Union[DictStrNum, Number, Number] = 1.0,
        resp_rate: Union[DictStrNum, Number] = 1.0,
        number_strata: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        """calculate the sample allocation.

        Args:
            target (Union[dict[Any, Number], Number]): the expected proportion used to calculate
                the sample size. It can be a single number for non-stratified designs or if all the strata have the same targeted proportion. Use a dictionary for stratified designs.
            precision (Union[dict[Any, Number], Number]): level of precision or half confidence
                interval. It can be a single number for non-stratified designs or if all the strata have the same targeted proportion. Use a dictionary for stratified designs.
            deff (Union[dict[Any, float], float], optional): design effect. It can be a single
                number for non-stratified designs or if all the strata have the same targeted proportion. Use a dictionary for stratified designs.. Defaults to 1.0.
            resp_rate (Union[dict[Any, float], float], optional): expected response rate. It can
                be a single number for non-stratified designs or if all the strata have the same targeted proportion. Use a dictionary for stratified designs.. Defaults to 1.0.
            number_strata (Optional[int], optional): number of strata. Defaults to None.
            alpha (float, optional): level of significance. Defaults to 0.05.

        Raises:
            AssertionError: when a dictionary is provided for a non-stratified design.
            AssertionError: when the dictionaries have different keys.
        """

        is_target_dict = isinstance(target, dict)
        is_precision_dict = isinstance(precision, dict)
        is_deff_dict = isinstance(deff, dict)
        is_resp_rate_dict = isinstance(resp_rate, dict)

        number_dictionaries = is_target_dict + is_precision_dict + is_deff_dict + is_resp_rate_dict

        stratum : Optional[list[StringNumber]] = None
        if not self.stratification and (
            isinstance(target, dict)
            or isinstance(precision, dict)
            or isinstance(deff, dict)
            or isinstance(resp_rate, dict)
        ):
            raise AssertionError("No python dictionary needed for non-stratified sample.")
        elif (
            not self.stratification
            and isinstance(target, (int, float))
            and isinstance(precision, (int, float))
            and isinstance(deff, (int, float))
            and isinstance(resp_rate, (int, float))
        ):
            self.deff_c = deff
            self.target = target
            self.precision = precision
            self.resp_rate = resp_rate
        elif (
            self.stratification
            and isinstance(target, (int, float))
            and isinstance(precision, (int, float))
            and isinstance(deff, (int, float))
            and isinstance(resp_rate, (int, float))
        ):
            if number_strata is not None:
                stratum = ["_stratum_" + str(i) for i in range(1, number_strata + 1)]
                self.target = dict(zip(stratum, np.repeat(target, number_strata)))
                self.precision = dict(zip(stratum, np.repeat(precision, number_strata)))
                self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
                self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
            else:
                raise ValueError("Number of strata not specified!")
        elif self.stratification and number_dictionaries > 0:
            dict_number = 0
            for ll in [target, precision, deff, resp_rate]:
                if isinstance(ll, dict):
                    dict_number += 1
                    if dict_number == 1:
                        stratum = list(ll.keys())
                    elif dict_number > 0:
                        if stratum != list(ll.keys()):
                            raise AssertionError("Python dictionaries have different keys")
            number_strata = len(stratum) if stratum is not None else 0
            if not is_target_dict and isinstance(target, (int, float)) and stratum is not None:
                self.target = dict(zip(stratum, np.repeat(target, number_strata)))
            elif isinstance(target, dict) :  
                self.target = target
            if not is_precision_dict and isinstance(precision, (int, float)) and stratum is not None:
                self.precision = dict(zip(stratum, np.repeat(precision, number_strata)))
            elif isinstance(precision, dict) :
                self.precision = precision
            if not is_deff_dict and isinstance(deff, (int, float)) and stratum is not None:
                self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
            elif isinstance(deff, dict):
                self.deff_c = deff
            if not is_resp_rate_dict and isinstance(resp_rate, (int, float)) and stratum is not None:
                self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
            elif isinstance(resp_rate, dict):
                self.resp_rate = resp_rate

        target_values : Union[np.ndarray, Number]
        precision_values : Union[np.ndarray, Number]
        resp_rate_values : Union[np.ndarray, Number]
        if isinstance(self.target, dict) and isinstance(self.precision, dict) and isinstance(self.resp_rate,dict):
            target_values = np.array(list(self.target.values()))
            precision_values = np.array(list(self.precision.values()))
            resp_rate_values = np.array(list(self.resp_rate.values()))
        elif isinstance(self.target, (int, float)) and isinstance(self.precision, (int, float)) and isinstance(self.resp_rate, (int, float)):
            target_values = self.target
            precision_values = self.precision
            resp_rate_values = self.resp_rate
        else:
            raise TypeError("Wrong type for self.target, self.precision, or self.resp_rate")
 
        if self.parameter == "proportion" and (
            np.asarray(0 > target_values).any()
            or np.asarray(target_values > 1).any()
            or np.asarray(0 > precision_values).any()
            or np.asarray(precision_values > 1).all()
        ):
            raise ValueError("Proportion values must be between 0 and 1.")

        self.alpha = alpha

        samp_size: Union[DictStrNum, Number]
        if self.method == "wald":
            samp_size = self._calculate_wald(
                target=self.target, precision=self.precision, stratum=stratum
            )
        elif self.method == "fleiss":
            samp_size = self._calculate_fleiss(
                target=self.target, precision=self.precision, stratum=stratum
            )

        if np.asarray(0 < resp_rate_values).all() and np.asarray(resp_rate_values <= 1).all():
            if isinstance(samp_size, dict) and isinstance(self.resp_rate, dict):
                for s in samp_size:
                    samp_size[s] = math.ceil(samp_size[s] / self.resp_rate[s])
            elif isinstance(samp_size, (int, float)) and isinstance(self.resp_rate, (int, float)):
                samp_size = math.ceil(samp_size / self.resp_rate)

        else:
            raise ValueError("Response rates must be between 0 and 1 (proportion).")

        self.samp_size = samp_size

    def allocate(self) -> Any:
        pass

    def optimize(self) -> Any:
        pass

    def to_dataframe(self, col_names: Optional[list[str]] = None) -> pd.DataFrame:
        """Coverts the dictionaries to a pandas dataframe

        Args:
            col_names (list[str], optional): column names for the dataframe. Defaults to
                ["_stratum", "_target", "_precision", "_samp_size"].

        Raises:
            AssertionError: when sample size is not calculated.

        Returns:
            pd.DataFrame: output pandas dataframe.
        """

        if self.samp_size is None:
            raise AssertionError("No sample size calculated.")
        else:
            if self.stratification:
                return  formats.dict_to_dataframe(
                        ["_stratum", "_target", "_precision", "_samp_size"], self.target, self.precision, self.samp_size
                    )
            if not self.stratification:
                return formats.dict_to_dataframe(
                        ["_target", "_precision", "_samp_size"], self.target, self.precision, self.samp_size
                    )
