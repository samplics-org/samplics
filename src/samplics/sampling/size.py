"""Sample size calculation module 

"""

from typing import Any, Dict, Tuple, Union, Optional

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
        self, parameter: str = "proportion", method: str = "wald", stratification: str = False
    ) -> None:

        self.parameter = parameter.lower()
        self.method = method.lower()
        if self.method not in ("wald"):
            raise AssertionError("Sample size calculation method not valid.")

        self.stratification = stratification
        self.deff: Dict[Any, float] = {"__none__": 1}

        ## Output data
        samp_size: Dict[Any, int]

    def icc(self):
        pass

    def deff(self):
        pass

    def _allocate_wald(
        self, target: Dict[Any, Number], precision: Dict[Any, Number], stratum: Optional[Array],
    ) -> Dict[Any, int]:

        z_value = normal().ppf(1 - self.alpha / 2)

        samp_size = {}
        if self.stratification:
            for s in stratum:
                samp_size[s] = math.ceil(
                    self.deff[s] * z_value ** 2 * target[s] * (1 - target[s]) / precision[s] ** 2
                )
        else:
            samp_size["__none__"] = math.ceil(
                self.deff["__none__"]
                * z_value ** 2
                * target["__none__"]
                * (1 - target["__none__"])
                / precision["__none__"] ** 2
            )

        return samp_size

    def _allocate_fleiss(
        self,
        target: Union[Dict[Any, Number], Number],
        precision: Union[Dict[Any, Number], Number],
        stratum: Optional[Array],
    ) -> Dict[Any, int]:
        pass

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
            deff = {"__none__": deff}
            target = {"__none__": target}
            precision = {"__none__": precision}
            resp_rate = {"__none__": resp_rate}

        if self.stratification and number_dictionaries == 0:
            stratum = ["_stratum_" + str(i) for i in range(1, number_strata + 1)]
            target = dict(zip(stratum, np.repeat(target, number_strata)))
            precision = dict(zip(stratum, np.repeat(precision, number_strata)))
            deff = dict(zip(stratum, np.repeat(deff, number_strata)))
            resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))
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
                target = dict(zip(stratum, np.repeat(target, number_strata)))
            if not is_precision_dict:
                precision = dict(zip(stratum, np.repeat(precision, number_strata)))
            if not is_deff_dict:
                deff = dict(zip(stratum, np.repeat(deff, number_strata)))
            if not is_resp_rate_dict:
                resp_rate = dict(zip(stratum, np.repeat(resp_rate, number_strata)))

        self.alpha = alpha
        self.deff = deff

        if self.method == "wald":
            samp_size = self._allocate_wald(target=target, precision=precision, stratum=stratum)
        elif self.method == "fleiss":
            samp_size = self._allocate_fleiss(target=target, precision=precision, stratum=stratum)

        self.samp_size = samp_size

    def to_dataframe(self) -> pd.DataFrame:
        pass
