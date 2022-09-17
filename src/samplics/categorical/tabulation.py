"""Cross-tabulation module 

The module implements the cross-tabulation analysis.

"""

from __future__ import annotations

import itertools

from typing import Any, Optional, Union

from patsy import dmatrix

import numpy as np
import pandas as pd

from scipy.stats import chi2, f

from samplics.estimation import TaylorEstimator
from samplics.utils.basic_functions import set_variables_names
from samplics.utils.formats import concatenate_series_to_str, numpy_array, remove_nans
from samplics.utils.types import Array, Number, Series, StringNumber, SinglePSUEst


class Tabulation:
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
        self.type = "oneway"
        self.point_est: dict[str, dict[StringNumber, Number]] = {}
        self.stats: dict[str, dict[str, Number]] = {}
        self.stderror: dict[str, dict[str, Number]] = {}
        self.lower_ci: dict[str, dict[str, Number]] = {}
        self.upper_ci: dict[str, dict[str, Number]] = {}
        self.deff: dict[str, dict[str, Number]] = {}
        self.alpha: float = alpha
        self.ciprop_method: str = ciprop_method
        self.design_info: dict[str, Number] = {}
        self.vars_names: Union[str, list[str]] = []
        self.vars_levels: dict[str, list[StringNumber]] = {}

    def __repr__(self) -> str:
        return f"Tabulation(parameter={self.parameter}, alpha={self.alpha})"

    def __str__(self) -> str:

        if self.vars_names == []:
            return "No categorical variables to tabulate"
        else:
            tbl_head = f"Tabulation of {self.vars_names[0]}"
            tbl_subhead1 = f" Number of strata: {self.design_info['number_strata']}"
            tbl_subhead2 = f" Number of PSUs: {self.design_info['number_psus']}"
            tbl_subhead3 = f" Number of observations: {self.design_info['number_obs']}"
            tbl_subhead4 = f" Degrees of freedom: {self.design_info['degrees_of_freedom']:.2f}"

            return f"\n{tbl_head}\n{tbl_subhead1}\n{tbl_subhead2}\n{tbl_subhead3}\n{tbl_subhead4}\n\n {self.to_dataframe().to_string(index=False)}\n"

    def _estimate(
        self,
        var_of_ones: Array,
        var: pd.DataFrame,
        samp_weight: Array = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        fpc: Union[dict, float] = 1,
        deff: bool = False,
        coef_variation: bool = False,
        single_psu: Union[SinglePSUEst, dict[StringNumber, SinglePSUEst]] = SinglePSUEst.error,
        strata_comb: Optional[dict[Array, Array]] = None,
        remove_nan: bool = False,
    ) -> tuple[TaylorEstimator, list, int]:

        if remove_nan:
            excluded_units = var.isna().values.ravel()
            var_of_ones, samp_weight, stratum, psu, ssu = remove_nans(
                excluded_units, var_of_ones, samp_weight, stratum, psu, ssu
            )
            var = var.dropna().astype(str)
        else:
            var.fillna("nan", inplace=True)
        var = var.to_numpy().ravel()

        var_of_ones = numpy_array(var_of_ones)

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
                single_psu=single_psu,
                strata_comb=strata_comb,
                remove_nan=False,
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
                single_psu=single_psu,
                strata_comb=strata_comb,
                remove_nan=False,
            )
        else:
            raise ValueError("parameter must be 'count' or 'proportion'")

        return tbl_est, list(np.unique(var)), var_of_ones.shape[0]

    def tabulate(
        self,
        vars: Array,
        varnames: Optional[Union[str, list[str]]] = None,
        samp_weight: Optional[Union[Array, Number]] = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        # TODO: by: Optional[Array] = None,
        fpc: Union[dict, float] = 1,
        deff: bool = False,
        coef_variation: bool = False,
        single_psu: Union[SinglePSUEst, dict[StringNumber, SinglePSUEst]] = SinglePSUEst.error,
        strata_comb: Optional[dict[Array, Array]] = None,
        remove_nan: bool = False,
    ) -> None:
        """
        docstring
        """

        if vars is None:
            raise AssertionError("vars need to be an array-like object")

        vars_df = pd.DataFrame(numpy_array(vars))
        nb_vars = 1 if len(vars_df.shape) == 1 else vars_df.shape[1]

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
            samp_weight = np.ones(vars_df.shape[0])
        elif isinstance(samp_weight, (int, float)):
            samp_weight = np.repeat(samp_weight, vars_df.shape[0])
        else:
            samp_weight = numpy_array(samp_weight)

        if nb_vars == 1:
            tbl_est, var_levels, nb_obs = self._estimate(
                var_of_ones=np.ones(vars_df.shape[0]),
                var=vars_df,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
                deff=deff,
                coef_variation=coef_variation,
                single_psu=single_psu,
                strata_comb=strata_comb,
                remove_nan=remove_nan,
            )
            self.vars_levels[vars_names[0]] = var_levels
            if self.parameter == "count":
                self.point_est[vars_names[0]] = tbl_est.point_est
                self.stderror[vars_names[0]] = tbl_est.stderror
                self.lower_ci[vars_names[0]] = tbl_est.lower_ci
                self.upper_ci[vars_names[0]] = tbl_est.upper_ci
                self.deff[vars_names[0]] = {}  # todo: tbl_est.deff
            elif self.parameter == "proportion":
                self.point_est[vars_names[0]] = tbl_est.point_est
                self.stderror[vars_names[0]] = tbl_est.stderror
                self.lower_ci[vars_names[0]] = tbl_est.lower_ci
                self.upper_ci[vars_names[0]] = tbl_est.upper_ci
                self.deff[vars_names[0]] = {}  # todo: tbl_est.deff
        else:
            nb_obs = 0
            var_of_ones = np.ones(vars_df.shape[0])
            for k in range(0, nb_vars):
                tbl_est, var_levels, nb_obs = self._estimate(
                    var_of_ones=var_of_ones,
                    var=vars_df.iloc[:, k],
                    samp_weight=samp_weight,
                    stratum=stratum,
                    psu=psu,
                    ssu=ssu,
                    fpc=fpc,
                    deff=deff,
                    coef_variation=coef_variation,
                    single_psu=single_psu,
                    strata_comb=strata_comb,
                    remove_nan=remove_nan,
                )
                self.vars_levels[vars_names[k]] = var_levels
                if self.parameter == "count":
                    self.point_est[vars_names[k]] = tbl_est.point_est
                    self.stderror[vars_names[k]] = tbl_est.stderror
                    self.lower_ci[vars_names[k]] = tbl_est.lower_ci
                    self.upper_ci[vars_names[k]] = tbl_est.upper_ci
                    self.deff[vars_names[k]] = {}  # todo: tbl_est.deff
                elif self.parameter == "proportion":
                    self.point_est[vars_names[k]] = tbl_est.point_est
                    self.stderror[vars_names[k]] = tbl_est.stderror
                    self.lower_ci[vars_names[k]] = tbl_est.lower_ci
                    self.upper_ci[vars_names[k]] = tbl_est.upper_ci
                    self.deff[vars_names[k]] = {}  # todo: tbl_est.deff

        self.vars_names = vars_names
        self.design_info = {
            "number_strata": tbl_est.number_strata,
            "number_psus": tbl_est.number_psus,
            "number_obs": nb_obs,
            "design_effect": 0,
            "degrees_of_freedom": tbl_est.number_psus - tbl_est.number_strata,
        }

    def to_dataframe(
        self,
    ) -> pd.DataFrame:

        oneway_df = pd.DataFrame([])

        for var in self.vars_names:
            var_df = pd.DataFrame(np.repeat(var, len(self.vars_levels[var])), columns=["variable"])
            var_df["category"] = self.vars_levels[var]
            var_df[self.parameter] = list(self.point_est[var].values())
            var_df["stderror"] = list(self.stderror[var].values())
            var_df["lower_ci"] = list(self.lower_ci[var].values())
            var_df["upper_ci"] = list(self.upper_ci[var].values())
            var_df.sort_values(by=["variable", "category"], inplace=True)
            oneway_df = oneway_df.append(var_df)

        return oneway_df


def saturated_two_ways_model(varsnames: list[str]) -> str:
    """
    docstring
    """

    varsnames_temp = [str(x) for x in varsnames]
    main_effects = " + ".join(varsnames_temp)
    interactions = ":".join(varsnames_temp)

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
        self.type = "twoway"
        self.point_est: dict[str, dict[StringNumber, Number]] = {}
        self.stats: dict[str, dict[str, Number]] = {}
        self.stderror: dict[str, dict[str, Number]] = {}
        self.covariance: dict[StringNumber, dict[StringNumber, Number]] = {}
        self.lower_ci: dict[str, dict[str, Number]] = {}
        self.upper_ci: dict[str, dict[str, Number]] = {}
        self.deff: dict[str, dict[str, Number]] = {}
        self.alpha: float = alpha
        self.ciprop_method: str = ciprop_method
        self.design_info: dict[str, Number] = {}
        self.vars_names: Union[str, list[str]] = []
        self.vars_levels: dict[str, StringNumber] = {}
        self.row_levels: list[StringNumber] = []
        self.col_levels: list[StringNumber] = []

    def __repr__(self) -> str:
        return f"CrossTabulation(parameter={self.parameter}, alpha={self.alpha})"

    def __str__(self) -> str:

        if self.vars_names == []:
            return "No categorical variables to tabulate"
        else:
            tbl_head = f"Cross-tabulation of {self.vars_names[0]} and {self.vars_names[1]}"
            tbl_subhead1 = f" Number of strata: {self.design_info['number_strata']}"
            tbl_subhead2 = f" Number of PSUs: {self.design_info['number_psus']}"
            tbl_subhead3 = f" Number of observations: {self.design_info['number_obs']}"
            tbl_subhead4 = f" Degrees of freedom: {self.design_info['degrees_of_freedom']:.2f}"

            chisq_dist = f"chi2({self.stats['Pearson-Unadj']['df']})"
            f_dist = f"F({self.stats['Pearson-Adj']['df_num']:.2f}, {self.stats['Pearson-Adj']['df_den']:.2f}"

            pearson_unadj = f"Unadjusted - {chisq_dist}: {self.stats['Pearson-Unadj']['chisq_value']:.4f} with p-value of {self.stats['Pearson-Unadj']['p_value']:.4f}"
            pearson_adj = f"Adjusted - {f_dist}): {self.stats['Pearson-Adj']['f_value']:.4f}  with p-value of {self.stats['Pearson-Adj']['p_value']:.4f}"
            pearson_test = (
                f"Pearson (with Rao-Scott adjustment):\n\t{pearson_unadj}\n\t{pearson_adj}"
            )

            lr_unadj = f" Unadjusted - {chisq_dist}: {self.stats['LR-Unadj']['chisq_value']:.4f} with p-value of {self.stats['LR-Unadj']['p_value']:.4f}"
            lr_adj = f" Adjusted - {f_dist}): {self.stats['LR-Adj']['f_value']:.4f}  with p-value of {self.stats['LR-Adj']['p_value']:.4f}"
            lr_test = f" Likelihood ratio (with Rao-Scott adjustment):\n\t{lr_unadj}\n\t{lr_adj}"

            return f"\n{tbl_head}\n{tbl_subhead1}\n{tbl_subhead2}\n{tbl_subhead3}\n{tbl_subhead4}\n\n {self.to_dataframe().to_string(index=False)}\n\n{pearson_test}\n\n {lr_test}\n"

    def _extract_estimates(self, tbl_est, vars_levels) -> tuple[np.ndarray, np.ndarray]:

        levels = list(tbl_est.point_est.keys())
        missing_levels = vars_levels[~np.isin(vars_levels, levels)]

        if missing_levels.shape[0] > 0:
            for level in vars_levels:
                if level in missing_levels:
                    tbl_est.point_est[level] = 0.0
                    tbl_est.covariance[level] = {}
                    for ll in vars_levels:
                        tbl_est.covariance[level][ll] = 0.0
                else:
                    for ll in missing_levels:
                        tbl_est.covariance[level][ll] = 0.0
                        tbl_est.covariance[level] = dict(sorted(tbl_est.covariance[level].items()))

        _tbl_est_point_est = dict(sorted(tbl_est.point_est.items()))
        _tbl_est_covariance = dict(sorted(tbl_est.covariance.items()))
        return (
            np.array(list(_tbl_est_point_est.values())),
            pd.DataFrame.from_dict(_tbl_est_covariance, orient="index").to_numpy(),
        )

    def tabulate(
        self,
        vars: Array,
        varnames: Optional[Union[str, list[str]]] = None,
        samp_weight: Optional[Union[Array, Number]] = None,
        stratum: Optional[Array] = None,
        psu: Optional[Array] = None,
        ssu: Optional[Array] = None,
        # Todo: by: Optional[Array] = None,
        fpc: Union[dict, float] = 1,
        deff: bool = False,
        coef_variation: bool = False,
        single_psu: Union[SinglePSUEst, dict[StringNumber, SinglePSUEst]] = SinglePSUEst.error,
        strata_comb: Optional[dict[Array, Array]] = None,
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

        if varnames is None:
            prefix = "var"
        elif isinstance(varnames, str):
            prefix = varnames
        elif isinstance(varnames, list):
            prefix = varnames[0]
        else:
            raise AssertionError("varnames should be a string or a list of string")

        vars_names = set_variables_names(vars, varnames, prefix)

        if isinstance(vars, np.ndarray):
            vars = pd.DataFrame(vars)

        samp_weight = numpy_array(samp_weight)
        stratum = numpy_array(stratum)
        psu = numpy_array(psu)
        ssu = numpy_array(ssu)

        if remove_nan:
            vars_nans = vars.isna()
            excluded_units = vars_nans.iloc[:, 0] | vars_nans.iloc[:, 1]
            samp_weight, stratum, psu, ssu = remove_nans(
                excluded_units, samp_weight, stratum, psu, ssu
            )
            vars = vars.dropna()
        else:
            vars = vars.fillna("nan")

        vars = vars.astype(str)
        vars.columns = vars_names

        vars.reset_index(inplace=True, drop=True)
        vars.sort_values(by=vars_names, inplace=True)
        samp_weight = samp_weight[vars.index]
        stratum = stratum[vars.index] if stratum is not None else None
        psu = psu[vars.index] if psu is not None else None
        ssu = ssu[vars.index] if ssu is not None else None

        vars_names_str = ["var_" + str(x) for x in vars_names]
        two_way_full_model = saturated_two_ways_model(vars_names_str)
        # vars.sort_values(by=vars_names, inplace=True)
        row_levels = vars[vars_names[0]].unique()
        col_levels = vars[vars_names[1]].unique()

        both_levels = [row_levels, col_levels]
        vars_levels = pd.DataFrame([ll for ll in itertools.product(*both_levels)])
        vars_levels.columns = vars_names_str

        vars_dummies = np.asarray(dmatrix(two_way_full_model, vars_levels, NA_action="raise"))

        if len(vars.shape) == 2:
            vars_for_oneway = np.apply_along_axis(
                func1d=concatenate_series_to_str, axis=1, arr=vars
            )
        else:
            vars_for_oneway = vars

        vars_levels_concat = np.apply_along_axis(
            func1d=concatenate_series_to_str, axis=1, arr=vars_levels
        )

        tbl_est_prop = TaylorEstimator(parameter="mean", alpha=self.alpha)
        tbl_est_prop.estimate(
            y=vars_for_oneway,
            samp_weight=samp_weight,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
            coef_variation=coef_variation,
            single_psu=single_psu,
            strata_comb=strata_comb,
            as_factor=True,
        )

        tbl_est = tbl_est_prop
        cell_est, cov_prop = self._extract_estimates(
            tbl_est=tbl_est_prop, vars_levels=vars_levels_concat
        )
        cov_prop_srs = (
            np.diag(cell_est)
            # - cell_est.reshape(vars_levels.shape[0], 1)
            - cell_est.reshape(cell_est.shape[0], 1)
            # @ np.transpose(cell_est.reshape(vars_levels.shape[0], 1))
            @ np.transpose(cell_est.reshape(cell_est.shape[0], 1))
        ) / vars.shape[0]

        if self.parameter == "count":
            tbl_est_count = TaylorEstimator(parameter="total", alpha=self.alpha)
            tbl_est_count.estimate(
                y=vars_for_oneway,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
                coef_variation=coef_variation,
                single_psu=single_psu,
                strata_comb=strata_comb,
                as_factor=True,
            )
            tbl_est = tbl_est_count

        nrows = row_levels.__len__()
        ncols = col_levels.__len__()
        x1 = vars_dummies[:, 1 : (nrows - 1) + (ncols - 1) + 1]  # main_effects
        x2 = vars_dummies[:, (nrows - 1) + (ncols - 1) + 1 :]  # interactions
        x1_t = np.transpose(x1)
        # breakpoint()
        x2_tilde = x2 - x1 @ np.linalg.inv(x1_t @ cov_prop_srs @ x1) @ (x1_t @ cov_prop_srs @ x2)
        delta_est = np.linalg.inv(np.transpose(x2_tilde) @ cov_prop_srs @ x2_tilde) @ (
            np.transpose(x2_tilde) @ cov_prop @ x2_tilde
        )
        tbl_keys = list(tbl_est.point_est.keys())
        cell_est = np.zeros(vars_levels.shape[0])
        cell_stderror = np.zeros(vars_levels.shape[0])
        cell_lower_ci = np.zeros(vars_levels.shape[0])
        cell_upper_ci = np.zeros(vars_levels.shape[0])
        for k in range(vars_levels.shape[0]):
            if vars_levels_concat[k] in tbl_keys:
                cell_est[k] = tbl_est.point_est[vars_levels_concat[k]]
                cell_stderror[k] = tbl_est.stderror[vars_levels_concat[k]]
                cell_lower_ci[k] = tbl_est.lower_ci[vars_levels_concat[k]]
                cell_upper_ci[k] = tbl_est.upper_ci[vars_levels_concat[k]]

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

        point_est_df = pd.DataFrame.from_dict(self.point_est, orient="index").values

        if self.parameter == "count":
            point_est_df = point_est_df / np.sum(point_est_df)
        point_est_null = point_est_df.sum(axis=1).reshape(nrows, 1) @ np.transpose(
            point_est_df.sum(axis=0).reshape(ncols, 1)
        )

        chisq_p = float(
            vars.shape[0] * np.sum((point_est_df - point_est_null) ** 2 / point_est_null)
        )
        f_p = float(chisq_p / np.trace(delta_est))

        chisq_lr = float(
            2 * vars.shape[0] * np.sum(point_est_df * np.log(point_est_df / point_est_null))
        )
        f_lr = float(chisq_lr / np.trace(delta_est))

        df_num = float((np.trace(delta_est) ** 2) / np.trace(delta_est @ delta_est))
        df_den = float((tbl_est.number_psus - tbl_est.number_strata) * df_num)

        self.stats = {
            "Pearson-Unadj": {
                "df": (nrows - 1) * (ncols - 1),
                "chisq_value": chisq_p,
                "p_value": 1 - chi2.cdf(chisq_p, (nrows - 1) * (ncols - 1)),
            },
            "Pearson-Adj": {
                "df_num": df_num,
                "df_den": df_den,
                "f_value": f_p,
                "p_value": 1 - f.cdf(f_p, df_num, df_den),
            },
            "LR-Unadj": {
                "df": (nrows - 1) * (ncols - 1),
                "chisq_value": chisq_lr,
                "p_value": 1 - chi2.cdf(chisq_lr, (nrows - 1) * (ncols - 1)),
            },
            "LR-Adj": {
                "df_num": df_num,
                "df_den": df_den,
                "f_value": f_lr,
                "p_value": 1 - f.cdf(f_lr, df_num, df_den),
            },
        }

        self.row_levels = list(row_levels)
        self.col_levels = list(col_levels)
        self.vars_names = vars_names
        self.design_info = {
            "number_strata": tbl_est.number_strata,
            "number_psus": tbl_est.number_psus,
            "number_obs": vars.shape[0],
            "design_effect": 0,
            "degrees_of_freedom": tbl_est.number_psus - tbl_est.number_strata,
        }

    def to_dataframe(
        self,
    ) -> pd.DataFrame:

        both_levels = [self.row_levels, self.col_levels]
        twoway_df = pd.DataFrame([ll for ll in itertools.product(*both_levels)])
        twoway_df.columns = self.vars_names

        for _ in range(len(self.row_levels)):
            for _ in range(len(self.col_levels)):
                twoway_df[self.parameter] = sum(
                    pd.DataFrame.from_dict(self.point_est, orient="index").values.tolist(), []
                )
                twoway_df["stderror"] = sum(
                    pd.DataFrame.from_dict(self.stderror, orient="index").values.tolist(), []
                )
                twoway_df["lower_ci"] = sum(
                    pd.DataFrame.from_dict(self.lower_ci, orient="index").values.tolist(), []
                )
                twoway_df["upper_ci"] = sum(
                    pd.DataFrame.from_dict(self.upper_ci, orient="index").values.tolist(), []
                )
        twoway_df.sort_values(by=self.vars_names, inplace=True)

        return twoway_df
