"""Cross-tabulation module 

The module implements the cross-tabulation analysis.

"""


from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.core.numeric import base_repr
import pandas as pd

from scipy.stats import t as student
from scipy.stats.stats import f_oneway

from samplics.utils.formats import numpy_array, remove_nans
from samplics.utils.types import Array, Number, StringNumber

from samplics.estimation import TaylorEstimator


class OneWay:
    def __init__(
        self,
        parameter: str = "count",
        alpha: float = 0.05,
        ciprop_method: str = "logit",
    ) -> None:

        if parameter.lower() not in ("count", "prop", "proportion"):
            self.parameter = parameter.lower()
        else:
            raise ValueError("parameter must be 'count' or 'proportion'")
        self.table_type = "oneway"
        self.table = Dict[StringNumber, StringNumber]
        self.stats = Dict[str, Number]
        # self.design = Dict[str, Number]
        self.alpha = alpha
        self.ciprop_method = ciprop_method

    # def _tabulate()

    def tabulate(
        self,
        vars: Array,
        varnames: Optional[List[str]] = None,
        samp_weight: Optional[Union[Array]] = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        domain: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
        deff: bool = False,
        coef_variation: bool = False,
        remove_nan: bool = False,
    ) -> None:
        """
        docstring
        """

        vars_np = numpy_array(vars)
        nb_vars = 1 if vars_np.shape[1] is None else vars_np.shape[1]

        if self.parameter == "proportion":
            pass
        elif self.parameter == "count":
            for _ in range(0, nb_vars):
                count_est = TaylorEstimator(
                    parameter=self.parameter, alpha=self.alpha, ciprop_method=self.ciprop_method
                )
                count_est.estimate(
                    vars_np[0], samp_weight, stratum, psu, ssu, domain, fpc, deff, coef_variation
                )
                breakpoint()

        else:
            raise ValueError("parameter must be 'count' or 'proportion'")


class TwoWay:
    """provides methods for analyzing cross-tabulations"""

    def __init__(self, table_type: str, alpha: float = 0.05) -> None:
        """[summary]

        Args:
            table_type (str): a string to indicate the type of the tabulation that is 'oneway'
                or 'twoway'.
            alpha (float): significant level for the confidence intervals
        """

        if table_type.lower() not in ("oneway", "twoway"):
            raise ValueError("table parameter must take values 'oneway' or 'twoway'!")

        self.table_type = table_type.lower()
        self.table = Dict[StringNumber, StringNumber]
        self.stats = Dict[str, Number]
        # self.design = Dict[str, Number]
        self.alpha = alpha

    def _oneway(
        self,
        row_var: Array,
        samp_weight: Array,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
    ) -> None:

        levels, counts = np.unique(row_var, return_counts=True)

        tbl_estimator = TaylorEstimator(parameter="total", alpha=self.alpha)
        tbl_estimator.estimate(
            y=np.ones(row_var.shape[0]),
            samp_weight=samp_weight,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            domain=row_var,
        )

    def _twoway():
        pass

    def tabulate(
        self,
        cat_vars: Array,  # Maybe a tuple or list of variables. If more than two than it can do all the possible 2x2 combinations
        samp_weight: Array,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
        remove_nan: bool = False,
    ) -> None:

        cat_vars = numpy_array(cat_vars)

        if self.table_type == "oneway" and cat_vars.shape[0] > 1:
            raise AssertionError("")

        samp_weight = numpy_array(samp_weight)
        if stratum is not None:
            stratum = numpy_array(stratum)
        if psu is not None:
            psu = numpy_array(psu)
        if ssu is not None:
            ssu = numpy_array(ssu)
        if fpc is not None:
            fpc = numpy_array(fpc)

        if remove_nan:
            excluded_units = np.isnan(samp_weight)
            samp_weight, stratum, psu, ssu, fpc = remove_nans(
                excluded_units, samp_weight, stratum, psu, ssu, fpc
            )

        if self.table_type == "oneway":
            self._oneway()
        elif self.table_type == "twoway":
            self._twoway()
        else:
            raise ValueError("Parameter 'table_type' is not valid!")