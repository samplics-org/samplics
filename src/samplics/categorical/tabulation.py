"""Cross-tabulation module 

The module implements the cross-tabulation analysis.

"""


from typing import Any, Dict, List, Optional, Union, Tuple

import itertools

import numpy as np
import pandas as pd

from patsy import dmatrix
from scipy.stats import chi2, f

from samplics.utils.basic_functions import set_variables_names
from samplics.utils.formats import (
    concatenate_series_to_str,
    numpy_array,
    numpy_to_dummies,
    remove_nans,
)
from samplics.utils.types import Array, Number, Series

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

        # breakpoint()

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


def saturated_two_ways_model(varsnames: List[str]) -> str:
    """
    docstring
    """

    main_effects = " + ".join(varsnames)
    interactions = ":".join(varsnames)

    return " + ".join([main_effects, interactions])


class CrossTabulation:
    """provides methods for analyzing cross-tabulations"""

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
        self.table_type = "twoway"
        self.point_est = {}  # Dict[str, Dict[StringNumber, Number]]
        self.stats = {}  # Dict[str, Dict[str, Number]]
        self.stderror = {}  # Dict[str, Dict[str, Number]]
        self.lower_ci = {}  # Dict[str, Dict[str, Number]]
        self.upper_ci = {}  # Dict[str, Dict[str, Number]]
        self.deff = {}  # Dict[str, Dict[str, Number]]
        self.alpha = alpha
        self.ciprop_method = ciprop_method

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

        if vars is None:
            raise AssertionError("vars need to be an array-like object")
        elif not isinstance(vars, (np.ndarray, pd.DataFrame)):
            vars = numpy_array(vars)

        if samp_weight is None:
            samp_weight = np.ones(vars.shape[0])
        elif isinstance(samp_weight, (int, float)):
            samp_weight = np.repeat(samp_weight, vars.shape[0])
        else:
            samp_weight = numpy_array(samp_weight)

        if isinstance(vars, np.ndarray):
            vars = pd.DataFrame(vars)

        if varnames is None:
            prefix = "var"
        elif isinstance(varnames, str):
            prefix = varnames
        elif isinstance(varnames, list):
            prefix = varnames[0]
        else:
            raise AssertionError("varnames should be a string or a list of string")

        if remove_nan:
            vars_nans = vars.isna()
            excluded_units = vars_nans.iloc[:, 0] | vars_nans.iloc[:, 1]
            samp_weight, stratum, psu, ssu = remove_nans(
                excluded_units, samp_weight, stratum, psu, ssu
            )
            vars.dropna(inplace=True)

        vars = vars.astype(str)
        vars_names = set_variables_names(vars, varnames, prefix)
        two_way_full_model = saturated_two_ways_model(vars_names)
        vars.columns = vars_names
        row_levels = vars[vars_names[0]].unique()
        col_levels = vars[vars_names[1]].unique()

        both_levels = [row_levels, col_levels]
        vars_levels = pd.DataFrame([ll for ll in itertools.product(*both_levels)])
        vars_levels.columns = vars_names

        # vars_levels = {vars_names[0]: row_levels, vars_names[1]: col_levels}
        # vars_levels = vars.drop_duplicates()
        # vars_levels.sort_values(by=vars_names, inplace=True)

        vars_dummies = np.asarray(dmatrix(two_way_full_model, vars_levels, NA_action="raise"))
        # breakpoint()

        if len(vars.shape) == 2:
            vars_for_oneway = np.apply_along_axis(
                func1d=concatenate_series_to_str, axis=1, arr=vars
            )
        else:
            vars_for_oneway = vars

        vars_levels_concat = np.apply_along_axis(
            func1d=concatenate_series_to_str, axis=1, arr=vars_levels
        )

        if self.parameter == "count":
            tbl_est_srs = TaylorEstimator(parameter="total", alpha=self.alpha)
            tbl_est_srs.estimate(
                y=np.ones(vars_for_oneway.shape[0]),
                samp_weight=np.ones(vars_for_oneway.shape[0]),
                domain=vars_for_oneway,
                fpc=fpc,
            )

            tbl_est = TaylorEstimator(parameter="total", alpha=self.alpha)
            tbl_est.estimate(
                y=np.ones(vars_for_oneway.shape[0]),
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                domain=vars_for_oneway,
                fpc=fpc,
            )
            # cell_est = np.asarray(list(tbl_est.point_est.values()))
            # cell_stderror = np.asarray(list(tbl_est.stderror.values()))
            # cell_lower_ci = np.asarray(list(tbl_est.lower_ci.values()))
            # cell_upper_ci = np.asarray(list(tbl_est.upper_ci.values()))

            tbl_keys = list(tbl_est_srs.point_est.keys())
            cell_est_srs = np.zeros(vars_levels.shape[0])
            # cell_stderror_srs = np.zeros(vars_levels.shape[0])
            cell_est = np.zeros(vars_levels.shape[0])
            cell_stderror = np.zeros(vars_levels.shape[0])
            cell_lower_ci = np.zeros(vars_levels.shape[0])
            cell_upper_ci = np.zeros(vars_levels.shape[0])
            for k in range(vars_levels.shape[0]):
                if vars_levels_concat[k] in tbl_keys:
                    cell_est_srs[k] = tbl_est_srs.point_est[vars_levels_concat[k]]
                    # cell_stderror_srs[k] = tbl_est_srs.stderror[vars_levels_concat[k]]

                    cell_est[k] = tbl_est.point_est[vars_levels_concat[k]]
                    cell_est[k] = tbl_est.point_est[vars_levels_concat[k]]
                    cell_stderror[k] = tbl_est.stderror[vars_levels_concat[k]]
                    cell_lower_ci[k] = tbl_est.lower_ci[vars_levels_concat[k]]
                    cell_upper_ci[k] = tbl_est.upper_ci[vars_levels_concat[k]]

        elif self.parameter == "proportion":
            tbl_est_srs = TaylorEstimator(parameter=self.parameter, alpha=self.alpha)
            tbl_est_srs.estimate(
                y=vars_for_oneway,
                samp_weight=np.ones(vars_for_oneway.shape[0]),
                fpc=fpc,
            )

            tbl_est = TaylorEstimator(parameter=self.parameter, alpha=self.alpha)
            tbl_est.estimate(
                y=vars_for_oneway,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
            )
            # cell_est = np.asarray(list(tbl_est.point_est["__none__"].values()))
            # cell_stderror = np.asarray(list(tbl_est.stderror["__none__"].values()))
            # cell_lower_ci = np.asarray(list(tbl_est.lower_ci["__none__"].values()))
            # cell_upper_ci = np.asarray(list(tbl_est.upper_ci["__none__"].values()))

            tbl_keys = list(tbl_est_srs.point_est["__none__"].keys())
            cell_est_srs = np.zeros(vars_levels.shape[0])
            # cell_stderror_srs = np.zeros(vars_levels.shape[0])
            cell_est = np.zeros(vars_levels.shape[0])
            cell_stderror = np.zeros(vars_levels.shape[0])
            cell_lower_ci = np.zeros(vars_levels.shape[0])
            cell_upper_ci = np.zeros(vars_levels.shape[0])
            for k in range(vars_levels.shape[0]):
                if vars_levels_concat[k] in tbl_keys:
                    cell_est_srs[k] = tbl_est_srs.point_est["__none__"][vars_levels_concat[k]]
                    # cell_stderror_srs[k] = tbl_est_srs.stderror["__none__"][vars_levels_concat[k]]

                    cell_est[k] = tbl_est.point_est["__none__"][vars_levels_concat[k]]
                    cell_stderror[k] = tbl_est.stderror["__none__"][vars_levels_concat[k]]
                    cell_lower_ci[k] = tbl_est.lower_ci["__none__"][vars_levels_concat[k]]
                    cell_upper_ci[k] = tbl_est.upper_ci["__none__"][vars_levels_concat[k]]
        else:
            raise ValueError("parameter must be 'count' or 'proportion'")

        cov_srs = (
            np.diag(cell_est_srs)
            - cell_est_srs.reshape(vars_levels.shape[0], 1)
            @ np.transpose(cell_est_srs.reshape(vars_levels.shape[0], 1))
        ) / (vars.shape[0] - 1)

        cov = (
            np.diag(cell_est)
            - cell_est.reshape(vars_levels.shape[0], 1)
            @ np.transpose(cell_est.reshape(vars_levels.shape[0], 1))
        ) / (vars.shape[0] - 1)

        nrows = row_levels.__len__()
        ncols = col_levels.__len__()
        x1 = vars_dummies[:, 1 : (nrows - 1) + (ncols - 1) + 1]  # main_effects
        x2 = vars_dummies[:, (nrows - 1) + (ncols - 1) + 1 :]  # interactions
        x1_t = np.transpose(x1)
        x2_tilde = x2 - x1 @ np.linalg.inv(x1_t @ cov_srs @ x1) @ (x1_t @ cov_srs @ x2)
        delta_est = np.linalg.inv(np.transpose(x2_tilde) @ cov_srs @ x2_tilde) @ (
            np.transpose(x2_tilde) @ cov @ x2_tilde
        )

        cov_srs = np.zeros((nrows, ncols))
        for r in range(nrows):
            point_est = {}
            stderror = {}
            lower_ci = {}
            upper_ci = {}
            for c in range(ncols):
                point_est.update(
                    {
                        vars_levels.iloc[r * ncols + c, 1]: cell_est[r * ncols + c],
                    }
                )
                stderror.update(
                    {
                        vars_levels.iloc[r * ncols + c, 1]: cell_stderror[r * ncols + c],
                    }
                )
                lower_ci.update(
                    {
                        vars_levels.iloc[r * ncols + c, 1]: cell_lower_ci[r * ncols + c],
                    }
                )
                upper_ci.update(
                    {
                        vars_levels.iloc[r * ncols + c, 1]: cell_upper_ci[r * ncols + c],
                    }
                )

            self.point_est.update({vars_levels.iloc[r * ncols, 0]: point_est})
            self.stderror.update({vars_levels.iloc[r * ncols, 0]: stderror})
            self.lower_ci.update({vars_levels.iloc[r * ncols, 0]: lower_ci})
            self.upper_ci.update({vars_levels.iloc[r * ncols, 0]: upper_ci})

        point_est = pd.DataFrame.from_dict(self.point_est, orient="index")

        point_est_null = point_est.sum(axis=1).values.reshape(nrows, 1) @ np.transpose(
            point_est.sum(axis=0).values.reshape(ncols, 1)
        )

        chisq_p = vars.shape[0] * np.sum((point_est.values - point_est_null) ** 2 / point_est_null)
        f_p = ((vars.shape[0] - 1) / vars.shape[0]) * chisq_p / np.trace(delta_est)

        df_num = np.trace(delta_est) ** 2 / np.trace(delta_est * delta_est)
        df_den = (tbl_est.number_psus - tbl_est.number_strata) * df_num

        self.stats = {
            "Pearson-Chisq": {
                "df": (nrows - 1) * (ncols - 1),
                "chisq_p": chisq_p,
                "p-value": chi2.pdf(chisq_p, (nrows - 1) * (ncols - 1)),
            },
            "Pearson-F": {
                "df_num": df_num,
                "df_den": df_den,
                "F_p": f_p,
                "p-value": f.pdf(f_p, df_num, df_den),
            },
        }

        breakpoint()
