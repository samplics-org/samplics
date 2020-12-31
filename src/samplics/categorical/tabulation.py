"""Cross-tabulation module 

The module implements the cross-tabulation analysis.

"""


from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.core.numeric import base_repr
import pandas as pd

from scipy.stats import t as student
from scipy.stats.stats import f_oneway

from samplics.utils.basic_functions import set_variables_names
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

        if parameter.lower() in ("count", "proportion"):
            self.parameter = parameter.lower()
        else:
            raise ValueError("parameter must be 'count' or 'proportion'")
        self.table_type = "oneway"
        self.table = {}  # Dict[str, Dict[StringNumber, Number]]
        self.stats = {}  # Dict[str, Dict[str, Number]]
        self.stderror = {}  # Dict[str, Dict[str, Number]]
        self.lower_ci = {}  # Dict[str, Dict[str, Number]]
        self.upper_ci = {}  # Dict[str, Dict[str, Number]]
        self.deff = {}  # Dict[str, Dict[str, Number]]
        self.alpha = alpha
        self.ciprop_method = ciprop_method

    def _estimate(
        self,
        var_of_ones: Array,
        var: Array,
        samp_weight: Array = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
        deff: bool = False,
        coef_variation: bool = False,
        remove_nan: bool = False,
    ) -> TaylorEstimator:

        if remove_nan:
            excluded_units = np.isnan(var) | np.isnan(samp_weight)
            var_of_ones, samp_weight, stratum, var, psu, ssu = remove_nans(
                excluded_units, var_of_ones, samp_weight, stratum, var, psu, ssu
            )

        if self.parameter == "count":
            tbl_est = TaylorEstimator(parameter="total", alpha=self.alpha)
            tbl_est.estimate(
                y=var_of_ones,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                domain=var,
                fpc=fpc,
                deff=deff,
                coef_variation=coef_variation,
                remove_nan=remove_nan,
            )
        elif self.parameter == "proportion":
            tbl_est = TaylorEstimator(parameter=self.parameter, alpha=self.alpha)
            tbl_est.estimate(
                y=var,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
                deff=deff,
                coef_variation=coef_variation,
                remove_nan=remove_nan,
            )
        else:
            raise ValueError("parameter must be 'count' or 'proportion'")

        return tbl_est

    def tabulate(
        self,
        vars: Array,
        varnames: Optional[List[str]] = None,
        samp_weight: Optional[Union[Array, Number]] = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        # Todo: by: Optional[Array] = None,
        fpc: Union[Dict, float] = 1,
        deff: bool = False,
        coef_variation: bool = False,
        remove_nan: bool = False,
    ) -> None:
        """
        docstring
        """

        if vars is None:
            raise AssertionError("vars need to be an array-like object")

        vars_np = numpy_array(vars)
        nb_vars = 1 if len(vars_np.shape) == 1 else vars_np.shape[1]
        # breakpoint()

        if varnames is None:
            prefix = "var"
        elif isinstance(varnames, str):
            prefix = varnames
        elif isinstance(varnames, list):
            prefix = varnames[0]
        else:
            raise AssertionError("varnames should be a string or a list of string")

        vars_names = set_variables_names(vars, varnames, prefix)

        if len(vars_names) != nb_vars:
            raise AssertionError(
                "Length of varnames must be the same as the number of columns of vars"
            )

        if samp_weight is None:
            samp_weight = np.ones(vars_np.shape[0])
        elif isinstance(samp_weight, (int, float)):
            samp_weight = np.repeat(samp_weight, vars_np.shape[0])
        else:
            samp_weight = numpy_array(samp_weight)

        if nb_vars == 1:
            tbl_est = self._estimate(
                var_of_ones=np.ones(vars_np.shape[0]),
                var=vars_np,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
                deff=deff,
                coef_variation=coef_variation,
                remove_nan=remove_nan,
            )
            if self.parameter == "count":
                self.stderror[vars_names[0]] = tbl_est.stderror
                self.table[vars_names[0]] = tbl_est.point_est
                self.lower_ci[vars_names[0]] = tbl_est.lower_ci
                self.upper_ci[vars_names[0]] = tbl_est.upper_ci
                self.deff[vars_names[0]] = {}  # todo: tbl_est.deff
            elif self.parameter == "proportion":
                self.stderror[vars_names[0]] = tbl_est.stderror["__none__"]
                self.table[vars_names[0]] = tbl_est.point_est["__none__"]
                self.lower_ci[vars_names[0]] = tbl_est.lower_ci["__none__"]
                self.upper_ci[vars_names[0]] = tbl_est.upper_ci["__none__"]
                self.deff[vars_names[0]] = {}  # todo: tbl_est.deff["__none__"]

            # breakpoint()
        else:
            var_of_ones = np.ones(vars_np.shape[0])
            for k in range(0, nb_vars):
                tbl_est = self._estimate(
                    var_of_ones=var_of_ones,
                    var=vars_np[:, k],
                    samp_weight=samp_weight,
                    stratum=stratum,
                    psu=psu,
                    ssu=ssu,
                    fpc=fpc,
                    deff=deff,
                    coef_variation=coef_variation,
                    remove_nan=remove_nan,
                )
                if self.parameter == "count":
                    self.stderror[vars_names[k]] = tbl_est.stderror
                    self.table[vars_names[k]] = tbl_est.point_est
                    self.lower_ci[vars_names[k]] = tbl_est.lower_ci
                    self.upper_ci[vars_names[k]] = tbl_est.upper_ci
                    self.deff[vars_names[k]] = {}  # todo: tbl_est.deff
                elif self.parameter == "proportion":
                    self.stderror[vars_names[k]] = tbl_est.stderror["__none__"]
                    self.table[vars_names[k]] = tbl_est.point_est["__none__"]
                    self.lower_ci[vars_names[k]] = tbl_est.lower_ci["__none__"]
                    self.upper_ci[vars_names[k]] = tbl_est.upper_ci["__none__"]
                    self.deff[vars_names[k]] = {}  # todo: tbl_est.deff["__none__"]


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