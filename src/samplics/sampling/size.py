"""Sample size calculation module 

"""

from typing import Any, Dict, Tuple, List, Union, Optional, overload

import math

import numpy as np
import pandas as pd

from scipy.stats import norm as normal

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber


class SampleSize:
    """*SampleSize* implements sample size calculation methods
    """

    def __init__(
        self, parameter: str = "proportion", method: str = "wald", stratification: bool = False
    ) -> None:

        self.parameter = parameter.lower()
        self.method = method.lower()
        if self.method not in ("wald", "fleiss"):
            raise AssertionError("Sample size calculation method not valid.")

        self.stratification = stratification
        self.deff_c: Dict[Any, float] = {"__none__": 1.0}
        self.deff_w: Dict[Any, float] = {"__none__": 1.0}

        ## Output data
        samp_size: Dict[Any, int]

    def icc(self) -> Union[Dict[Any, Number], Number]:
        pass

    def deff(
        self, cluster_size: Union[Dict[Any, Number], Number], icc: Union[Dict[Any, Number], Number]
    ) -> Union[Dict[Any, Number], Number]:

        if isinstance(cluster_size, (int, float)) and isinstance(icc, (int, float)):
            return max(1 + (cluster_size - 1) * icc, 0)
        elif isinstance(cluster_size, dict) and isinstance(icc, dict):
            if cluster_size.keys() != icc.keys():
                raise AssertionError("Parameters do not have the same dictionary keys.")
            deff_c: Dict[Any, Number] = {}
            for s in cluster_size:
                deff_c[s] = max(1 + (cluster_size[s] - 1) * icc[s], 0)
            return deff_c
        else:
            raise ValueError("Combination of types not supported.")

    def _allocate_wald(
        self, target: Dict[Any, Number], precision: Dict[Any, Number], stratum: Optional[Array],
    ) -> Dict[Any, int]:

        z_value = normal().ppf(1 - self.alpha / 2)

        samp_size: Dict[Any, int] = {}
        if self.stratification:
            for s in stratum:
                samp_size[s] = math.ceil(
                    self.deff_c[s] * z_value ** 2 * target[s] * (1 - target[s]) / precision[s] ** 2
                )
        else:
            samp_size["__none__"] = math.ceil(
                self.deff_c["__none__"]
                * z_value ** 2
                * target["__none__"]
                * (1 - target["__none__"])
                / precision["__none__"] ** 2
            )

        return samp_size

    def _allocate_fleiss(
        self, target: Dict[Any, Number], precision: Dict[Any, Number], stratum: Optional[Array],
    ) -> Dict[Any, int]:

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

        samp_size = {}
        if self.stratification:
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
        else:
            fct = fleiss_factor(target["__none__"], precision["__none__"])
            samp_size["__none__"] = math.ceil(
                self.deff_c["__none__"]
                * (
                    fct * (z_value ** 2) / (4 * precision["__none__"] ** 2)
                    + 1 / precision["__none__"]
                    - 2 * z_value ** 2
                    + (z_value + 2) / fct
                )
            )

        return samp_size

    def allocate(
        self,
        target: Union[Dict[Any, Number], Number],
        precision: Union[Dict[Any, Number], Number],
        deff: Union[Dict[Any, float], float] = 1.0,
        resp_rate: Union[Dict[Any, float], float] = 1.0,
        number_strata: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:

        is_target_dict = isinstance(target, dict)
        is_precision_dict = isinstance(precision, dict)
        is_deff_dict = isinstance(deff, dict)
        is_resp_rate_dict = isinstance(resp_rate, dict)

        number_dictionaries = is_target_dict + is_precision_dict + is_deff_dict + is_resp_rate_dict

        if not self.stratification and number_dictionaries > 0:
            raise AssertionError("No python dictionary needed for non-stratified sample.")
        elif not self.stratification and number_dictionaries == 0:
            stratum = None
            if isinstance(deff, float):  # not necssary for the logic but will make mypy happy
                self.deff_c = {"__none__": deff}
            if isinstance(target, (int, float)):
                self.target = {"__none__": target}
            if isinstance(precision, (int, float)):
                self.precision = {"__none__": precision}
            if isinstance(resp_rate, (int, float)):
                self.resp_rate = {"__none__": resp_rate}

        if self.stratification and number_dictionaries == 0:
            stratum = ["_stratum_" + str(i) for i in range(1, number_strata + 1)]
            self.target = dict(zip(stratum, np.repeat(target, number_strata)))
            self.precision = dict(zip(stratum, np.repeat(precision, number_strata)))
            self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
            self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
        elif self.stratification and number_dictionaries > 0:
            dict_number = 0
            for ll in [target, precision, deff, resp_rate]:
                if isinstance(ll, dict):
                    dict_number += 1
                    if dict_number == 1:
                        strata = ll.keys()
                    elif dict_number > 0:
                        if strata != ll.keys():
                            raise AssertionError("Python dictionaries have different keys")
            stratum = list(strata)
            number_strata = len(stratum)
            if not is_target_dict:
                self.target = dict(zip(stratum, np.repeat(target, number_strata)))
            elif isinstance(target, dict):  # to make mypy happy
                self.target = target
            if not is_precision_dict:
                self.precision = dict(zip(stratum, np.repeat(precision, number_strata)))
            elif isinstance(precision, dict):
                self.precision = precision
            if not is_deff_dict:
                self.deff_c = dict(zip(stratum, np.repeat(deff, number_strata)))
            elif isinstance(deff, dict):
                self.deff_c = deff
            if not is_resp_rate_dict:
                self.resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
            elif isinstance(resp_rate, dict):
                self.resp_rate = resp_rate

        self.alpha = alpha

        if self.method == "wald":
            samp_size = self._allocate_wald(
                target=self.target, precision=self.precision, stratum=stratum
            )
        elif self.method == "fleiss":
            samp_size = self._allocate_fleiss(
                target=self.target, precision=self.precision, stratum=stratum
            )

        self.samp_size = samp_size

    def to_dataframe(
        self, col_names: List[str] = ["_stratum", "_target", "_precision", "_samp_size"]
    ) -> pd.DataFrame:

        ncols = len(col_names)

        if self.samp_size is None:
            raise AssertionError("No sample size calculated.")
        else:
            samp_size_df = formats.dict_to_dataframe(
                col_names, self.target, self.precision, self.samp_size
            )

        return samp_size_df
