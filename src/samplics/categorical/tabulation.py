"""Cross-tabulation module

The module implements the cross-tabulation analysis.

"""

from __future__ import annotations

import itertools

from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from patsy import dmatrix
from scipy.stats import chi2, f

from samplics.estimation import TaylorEstimator
from samplics.utils.basic_functions import set_variables_names
from samplics.utils.errors import DimensionError
from samplics.utils.formats import numpy_array, remove_nans
from samplics.utils.types import Array, Number, PopParam, SinglePSUEst, StringNumber


class Tabulation:
    def __init__(
        self,
        param: PopParam,
        alpha: float = 0.05,
        ciprop_method: str = "logit",
    ) -> None:
        if param not in (PopParam.count, PopParam.prop):
            raise ValueError("Parameter must be 'count' or 'proportion'!")
        self.param = param
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
        return f"Tabulation(parameter={self.param}, alpha={self.alpha})"

    def __str__(self) -> str:
        if self.vars_names == []:
            return "No categorical variables to tabulate"
        else:
            tbl_head = f"Tabulation of {self.vars_names[0]}"
            tbl_subhead1 = f" Number of strata: {self.design_info['nb_strata']}"
            tbl_subhead2 = f" Number of PSUs: {self.design_info['nb_psus']}"
            tbl_subhead3 = f" Number of observations: {self.design_info['nb_obs']}"
            tbl_subhead4 = (
                f" Degrees of freedom: {self.design_info['degrees_of_freedom']:.2f}"
            )

            return f"\n{tbl_head}\n{tbl_subhead1}\n{tbl_subhead2}\n{tbl_subhead3}\n{tbl_subhead4}\n\n {self.to_dataframe().to_string(index=False)}\n"

    def _estimate(
        self,
        var_of_ones: Array,
        var: pd.DataFrame,
        samp_weight: np.ndarray,
        stratum: np.ndarray,
        psu: np.ndarray,
        ssu: np.ndarray,
        fpc: Union[dict, float] = 1,
        deff: bool = False,
        coef_var: bool = False,
        single_psu: Union[
            SinglePSUEst, dict[StringNumber, SinglePSUEst]
        ] = SinglePSUEst.error,
        strata_comb: Optional[dict[Array, Array]] = None,
        remove_nan: bool = False,
    ) -> tuple[TaylorEstimator, list, int]:
        if remove_nan:
            to_keep = remove_nans(
                var.values.ravel().shape[0],
                var_of_ones,
                samp_weight,
                stratum,
                psu,
                ssu,
            )
            if var.ndim == 1:  # Series
                to_keep = to_keep & remove_nans(
                    var.values.ravel().shape[0], var.values.ravel()
                )
            elif var.ndim == 2:  # DataFrame
                for col in var.columns:
                    to_keep = to_keep & remove_nans(
                        var.values.ravel().shape[0], var[col].values.ravel()
                    )
            else:
                raise DimensionError("The dimension must be 1 or 2.")

            var_of_ones = var_of_ones[to_keep]
            var = var.loc[to_keep]
            samp_weight = samp_weight[to_keep]
            stratum = stratum[to_keep] if stratum.shape not in ((), (0,)) else stratum
            psu = psu[to_keep] if psu.shape not in ((), (0,)) else psu
            ssu = ssu[to_keep] if ssu.shape not in ((), (0,)) else ssu
        else:
            var.fillna("nan", inplace=True)

        var_of_ones = numpy_array(var_of_ones)

        if self.param == PopParam.count:
            tbl_est = TaylorEstimator(param=PopParam.total, alpha=self.alpha)
            tbl_est.estimate(
                y=var_of_ones,
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                domain=var.to_numpy().ravel(),
                fpc=fpc,
                deff=deff,
                coef_var=coef_var,
                single_psu=single_psu,
                strata_comb=strata_comb,
                remove_nan=False,
            )
        elif self.param == PopParam.prop:
            tbl_est = TaylorEstimator(param=self.param, alpha=self.alpha)
            tbl_est.estimate(
                y=var.to_numpy().ravel(),
                samp_weight=samp_weight,
                stratum=stratum,
                psu=psu,
                ssu=ssu,
                fpc=fpc,
                deff=deff,
                coef_var=coef_var,
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
        coef_var: bool = False,
        single_psu: Union[
            SinglePSUEst, dict[StringNumber, SinglePSUEst]
        ] = SinglePSUEst.error,
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

        _samp_weight = numpy_array(samp_weight)

        _samp_weight = (
            np.ones(vars_df.shape[0])
            if _samp_weight.shape in ((), (0,))
            else _samp_weight
        )
        _samp_weight = (
            np.repeat(_samp_weight, vars_df.shape[0])
            if _samp_weight.shape[0] == 1
            else _samp_weight
        )
        _stratum = numpy_array(stratum)
        _psu = numpy_array(psu)
        _ssu = numpy_array(ssu)

        if _samp_weight.shape != ():
            positive_weights = _samp_weight > 0
            _samp_weight = _samp_weight[positive_weights]
            vars_df = vars_df[positive_weights]
            _stratum = _stratum[positive_weights] if _stratum.shape != () else _stratum
            _psu = _psu[positive_weights] if _psu.shape != () else _psu
            _ssu = _ssu[positive_weights] if _ssu.shape != () else _ssu

        if nb_vars == 1:
            tbl_est, var_levels, nb_obs = self._estimate(
                var_of_ones=np.ones(vars_df.shape[0]),
                var=vars_df,
                samp_weight=_samp_weight,
                stratum=_stratum,
                psu=_psu,
                ssu=_ssu,
                fpc=fpc,
                deff=deff,
                coef_var=coef_var,
                single_psu=single_psu,
                strata_comb=strata_comb,
                remove_nan=remove_nan,
            )
            self.vars_levels[vars_names[0]] = var_levels
            if self.param == PopParam.count:
                self.point_est[vars_names[0]] = tbl_est.point_est
                self.stderror[vars_names[0]] = tbl_est.stderror
                self.lower_ci[vars_names[0]] = tbl_est.lower_ci
                self.upper_ci[vars_names[0]] = tbl_est.upper_ci
                self.deff[vars_names[0]] = {}  # todo: tbl_est.deff
            elif self.param == PopParam.prop:
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
                    samp_weight=_samp_weight,
                    stratum=_stratum,
                    psu=_psu,
                    ssu=_ssu,
                    fpc=fpc,
                    deff=deff,
                    coef_var=coef_var,
                    single_psu=single_psu,
                    strata_comb=strata_comb,
                    remove_nan=remove_nan,
                )
                self.vars_levels[vars_names[k]] = var_levels
                if self.param == PopParam.count:
                    self.point_est[vars_names[k]] = tbl_est.point_est
                    self.stderror[vars_names[k]] = tbl_est.stderror
                    self.lower_ci[vars_names[k]] = tbl_est.lower_ci
                    self.upper_ci[vars_names[k]] = tbl_est.upper_ci
                    self.deff[vars_names[k]] = {}  # todo: tbl_est.deff
                elif self.param == PopParam.prop:
                    self.point_est[vars_names[k]] = tbl_est.point_est
                    self.stderror[vars_names[k]] = tbl_est.stderror
                    self.lower_ci[vars_names[k]] = tbl_est.lower_ci
                    self.upper_ci[vars_names[k]] = tbl_est.upper_ci
                    self.deff[vars_names[k]] = {}  # todo: tbl_est.deff

        self.vars_names = vars_names
        self.design_info = {
            "nb_strata": tbl_est.nb_strata,
            "nb_psus": tbl_est.nb_psus,
            "nb_obs": nb_obs,
            "design_effect": 0,
            "degrees_of_freedom": tbl_est.nb_psus - tbl_est.nb_strata,
        }

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        oneway_df = pd.DataFrame([])

        for var in self.vars_names:
            var_df = pd.DataFrame(
                np.repeat(var, len(self.vars_levels[var])), columns=["variable"]
            )
            var_df["category"] = self.vars_levels[var]
            var_df[self.param] = list(self.point_est[var].values())
            var_df["stderror"] = list(self.stderror[var].values())
            var_df["lower_ci"] = list(self.lower_ci[var].values())
            var_df["upper_ci"] = list(self.upper_ci[var].values())
            var_df.sort_values(by=["variable", "category"], inplace=True)
            oneway_df = pd.concat((oneway_df, var_df))

        return oneway_df


def _saturated_two_ways_model(varsnames: list[str]) -> str:
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
        param: str = PopParam.count,
        alpha: float = 0.05,
        ciprop_method: str = "logit",
    ) -> None:
        if param not in (PopParam.count, PopParam.prop):
            raise ValueError("Parameter must be 'count' or 'proportion'!")
        self.param = param
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
        return f"CrossTabulation(parameter={self.param}, alpha={self.alpha})"

    def __str__(self) -> str:
        if self.vars_names == []:
            return "No categorical variables to tabulate"
        else:
            tbl_head = (
                f"Cross-tabulation of {self.vars_names[0]} and {self.vars_names[1]}"
            )
            tbl_subhead1 = f" Number of strata: {self.design_info['nb_strata']}"
            tbl_subhead2 = f" Number of PSUs: {self.design_info['nb_psus']}"
            tbl_subhead3 = f" Number of observations: {self.design_info['nb_obs']}"
            tbl_subhead4 = (
                f" Degrees of freedom: {self.design_info['degrees_of_freedom']:.2f}"
            )

            chisq_dist = f"chi2({self.stats['Pearson-Unadj']['df']})"
            f_dist = f"F({self.stats['Pearson-Adj']['df_num']:.2f}, {self.stats['Pearson-Adj']['df_den']:.2f}"

            pearson_unadj = f"Unadjusted - {chisq_dist}: {self.stats['Pearson-Unadj']['chisq_value']:.4f} with p-value of {self.stats['Pearson-Unadj']['p_value']:.4f}"
            pearson_adj = f"Adjusted - {f_dist}): {self.stats['Pearson-Adj']['f_value']:.4f}  with p-value of {self.stats['Pearson-Adj']['p_value']:.4f}"
            pearson_test = f"Pearson (with Rao-Scott adjustment):\n\t{pearson_unadj}\n\t{pearson_adj}"

            lr_unadj = f" Unadjusted - {chisq_dist}: {self.stats['LR-Unadj']['chisq_value']:.4f} with p-value of {self.stats['LR-Unadj']['p_value']:.4f}"
            lr_adj = f" Adjusted - {f_dist}): {self.stats['LR-Adj']['f_value']:.4f}  with p-value of {self.stats['LR-Adj']['p_value']:.4f}"
            lr_test = f" Likelihood ratio (with Rao-Scott adjustment):\n\t{lr_unadj}\n\t{lr_adj}"

            return f"\n{tbl_head}\n{tbl_subhead1}\n{tbl_subhead2}\n{tbl_subhead3}\n{tbl_subhead4}\n\n {self.to_dataframe().to_string(index=False)}\n\n{pearson_test}\n\n {lr_test}\n"

    # also mutates tbl_est
    def _extract_estimates(
        self, tbl_est, vars_levels
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        levels = list(tbl_est.point_est.keys())
        missing_levels = vars_levels[~np.isin(vars_levels, levels)]
        if missing_levels.shape[0] > 0:
            for level in vars_levels:
                if level in missing_levels:
                    tbl_est.point_est[level] = 0.0
                    tbl_est.stderror[level] = 0.0
                    tbl_est.lower_ci[level] = 0.0
                    tbl_est.upper_ci[level] = 0.0
                    tbl_est.covariance[level] = {}
                    for ll in vars_levels:
                        tbl_est.covariance[level][ll] = 0.0
                else:
                    for ll in missing_levels:
                        tbl_est.covariance[level][ll] = 0.0
                    tbl_est.covariance[level] = dict(
                        sorted(tbl_est.covariance[level].items())
                    )

        _tbl_est_point_est = dict(sorted(tbl_est.point_est.items()))
        _tbl_est_covariance = dict(sorted(tbl_est.covariance.items()))
        return (
            np.array(list(_tbl_est_point_est.values())),
            pd.DataFrame.from_dict(_tbl_est_covariance, orient="index").to_numpy(),
            missing_levels,
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
        coef_var: bool = False,
        single_psu: Union[
            SinglePSUEst, dict[StringNumber, SinglePSUEst]
        ] = SinglePSUEst.error,
        strata_comb: Optional[dict[Array, Array]] = None,
        remove_nan: bool = False,
    ) -> None:
        if vars is None:
            raise AssertionError("vars need to be an array-like object")
        elif not isinstance(vars, (np.ndarray, pd.DataFrame)):
            vars = pl.DataFrame(numpy_array(vars))
            vars.columns = ["__var1__", "__var2__"]

        if samp_weight is None:
            samp_weight = np.ones(vars.shape[0])
        elif isinstance(samp_weight, (int, float)):
            samp_weight = np.repeat(samp_weight, vars.shape[0])
        else:
            samp_weight = numpy_array(samp_weight)

        # if varnames is None:
        #     prefix = "var"
        # elif isinstance(varnames, str):
        #     prefix = varnames
        # elif isinstance(varnames, list):
        #     prefix = varnames[0]
        # else:
        #     raise AssertionError("varnames should be a string or a list of string")

        # vars_names = set_variables_names(vars, varnames, prefix)

        if isinstance(vars, (np.ndarray, pd.DataFrame)):
            vars = pl.DataFrame(vars)

        if varnames is not None:
            vars.columns = varnames

        vars_names = vars.columns

        df = vars.with_columns(
            samp_weight=numpy_array(samp_weight),
            stratum=(
                np.repeat("__none__", vars.shape[0])
                if stratum is None
                else numpy_array(stratum)
            ),
            psu=(
                np.linspace(1, vars.shape[0], num=vars.shape[0])
                if psu is None
                else numpy_array(psu)
            ),
            ssu=np.repeat(1, vars.shape[0]) if ssu is None else numpy_array(ssu),
        ).filter(pl.col("samp_weight") > 0)

        df = df.cast(
            {
                vars_names[0]: pl.String,
                vars_names[1]: pl.String,
                "samp_weight": pl.Float64,
                "stratum": pl.String,
                "psu": pl.String,
                "ssu": pl.String,
            }
        )

        if remove_nan:
            df = (
                df.filter(
                    (
                        pl.col(vars_names[0]).is_not_null()
                        & ~pl.col(vars_names[1]).eq("NaN")
                    )
                    & (
                        pl.col(vars_names[1]).is_not_null()
                        & ~pl.col(vars_names[1]).eq("NaN")
                    )
                    & (
                        pl.col("samp_weight").is_not_null()
                        & pl.col("samp_weight").is_not_nan()
                    )
                    & (
                        pl.col("stratum").is_not_null()
                        & ~pl.col(vars_names[1]).eq("NaN")
                    )
                    & (pl.col("psu").is_not_null() & ~pl.col(vars_names[1]).eq("NaN"))
                    & (pl.col("ssu").is_not_null() & ~pl.col(vars_names[1]).eq("NaN"))
                )
                .with_columns(
                    pl.col(vars_names[0]).cast(pl.String),
                    pl.col(vars_names[1]).cast(pl.String),
                )
                .sort(vars_names)
            )
        else:
            df = df.with_columns(
                pl.col(vars_names[0]).fill_null("__null__"),
                pl.col(vars_names[1]).fill_null("__null__"),
            ).sort(vars_names)

        if len(df.shape) == 2:
            vars_for_oneway = (
                df.select(vars_names)
                .with_columns(
                    (pl.col(vars_names[0]) + "__by__" + pl.col(vars_names[1])).alias(
                        "__cross_vars__"
                    )
                )["__cross_vars__"]
                .to_numpy()
            )
        else:
            vars_for_oneway = df.to_series().to_numpy()

        param = PopParam.total if self.param == PopParam.count else PopParam.mean

        tbl_est = TaylorEstimator(param=param, alpha=self.alpha)
        tbl_est.estimate(
            y=vars_for_oneway,
            samp_weight=df["samp_weight"],
            stratum=df["stratum"].to_numpy(),
            psu=df["psu"].to_numpy(),
            ssu=None,
            fpc=fpc,
            coef_var=coef_var,
            single_psu=single_psu,
            strata_comb=strata_comb,
            as_factor=True,
        )
        if self.param == PopParam.count:
            tbl_est_prop = TaylorEstimator(param=PopParam.mean, alpha=self.alpha)
            tbl_est_prop.estimate(
                y=vars_for_oneway,
                samp_weight=df["samp_weight"],
                stratum=df["stratum"].to_numpy(),
                psu=df["psu"].to_numpy(),
                ssu=None,
                fpc=fpc,
                coef_var=coef_var,
                single_psu=single_psu,
                strata_comb=strata_comb,
                as_factor=True,
            )
            cell_est, cov_est, missing_levels = self._extract_estimates(
                tbl_est=tbl_est_prop, vars_levels=np.unique(vars_for_oneway)
            )
        elif self.param == PopParam.prop:
            cell_est, cov_est, missing_levels = self._extract_estimates(
                tbl_est=tbl_est, vars_levels=np.unique(vars_for_oneway)
            )
        else:
            raise ValueError("parameter must be 'count' or 'proportion'")

        cov_est_srs = (np.diag(cell_est) - cell_est.reshape((cell_est.shape[0], 1)) @ cell_est.reshape((1, cell_est.shape[0]))) / df.shape[0]
        # cov_est_srs = cov_est_srs * ((df.shape[0] - 1) / df.shape[0])

        two_way_full_model = _saturated_two_ways_model(vars_names)

        vars_dummies = np.asarray(
            dmatrix(
                two_way_full_model,
                df.select(vars_names).unique().sort(vars_names).to_pandas(),
                NA_action="raise",
            )
        )

        row_labels = df[vars_names[0]].unique().sort()
        col_labels = df[vars_names[1]].unique().sort()
        nrows = row_labels.len()
        ncols = col_labels.len()
        x1 = vars_dummies[:, 0 : (nrows - 1) + (ncols - 1) + 1]  # main_effects
        x2 = vars_dummies[:, (nrows - 1) + (ncols - 1) + 1 :]  # interactions

        # TODO:
        # Replace the inversion of cov_prop and cov_prop_srs below by the a multiplication
        # we have that inv(x' V x) = inv(x' L L' x) = z'z where z = inv(L) x
        # L is the Cholesky factor i.e. L = np.linalg.cholesky(V)

        try:
            x1_t = np.transpose(x1)
            x2_tilde = x2 - x1 @ np.linalg.inv(x1_t @ cov_est_srs @ x1) @ (
                x1_t @ cov_est_srs @ x2
            )
            delta_est = np.linalg.inv(
                np.transpose(x2_tilde) @ cov_est_srs @ x2_tilde
            ) @ (np.transpose(x2_tilde) @ cov_est @ x2_tilde)
        except np.linalg.LinAlgError:
            delta_est = np.zeros((nrows * ncols, nrows * ncols))

        keys = list(tbl_est.point_est.keys())
        for row in row_labels:
            for col in col_labels:
                key = f"{row}__by__{col}"
                if key not in keys:
                    tbl_est.point_est[key] = 0.0
                    tbl_est.stderror[key] = 0.0
                    tbl_est.lower_ci[key] = 0.0
                    tbl_est.upper_ci[key] = 0.0

        tbl_df = (
            pl.DataFrame(None)
            .with_columns(
                key=np.array(list(tbl_est.point_est.keys())),
                point_est=np.array(list(tbl_est.point_est.values())),
                stderror=np.array(list(tbl_est.stderror.values())),
                lower_ci=np.array(list(tbl_est.lower_ci.values())),
                upper_ci=np.array(list(tbl_est.upper_ci.values())),
            )
            .with_columns(pl.col("key").str.split("__by__"))
            .with_columns(
                pl.col("key").list.get(0).alias(vars_names[0]),
                pl.col("key").list.get(1).alias(vars_names[1]),
            )
            .drop("key")
        )

        poin_est_dict = tbl_df.select(vars_names + ["point_est"]).rows_by_key(
            key=vars_names[0]
        )
        stderror_dict = tbl_df.select(vars_names + ["stderror"]).rows_by_key(
            key=vars_names[0]
        )
        lower_ci_dict = tbl_df.select(vars_names + ["lower_ci"]).rows_by_key(
            key=vars_names[0]
        )
        upper_ci_dict = tbl_df.select(vars_names + ["upper_ci"]).rows_by_key(
            key=vars_names[0]
        )

        for var1 in poin_est_dict:
            point_est = {}
            stderror = {}
            lower_ci = {}
            upper_ci = {}
            for k, var2 in enumerate(poin_est_dict[var1]):
                point_est.update({var2[0]: var2[1]})
                stderror.update({var2[0]: stderror_dict[var1][k][1]})
                lower_ci.update({var2[0]: lower_ci_dict[var1][k][1]})
                upper_ci.update({var2[0]: upper_ci_dict[var1][k][1]})
            self.point_est[var1] = point_est
            self.stderror[var1] = stderror
            self.lower_ci[var1] = lower_ci
            self.upper_ci[var1] = upper_ci

        if self.param == PopParam.count:
            tbl_df = tbl_df.with_columns(
                (pl.col("point_est") / pl.col("point_est").sum()).alias("est_prop")
            )
        elif self.param == PopParam.prop:
            tbl_df = tbl_df.with_columns(est_prop=pl.col("point_est"))
        else:
            raise ValueError("parameter must be 'count' or 'proportion'")

        tbl_df = (
            tbl_df.join(
                other=tbl_df.group_by(vars_names[0]).agg(
                    pl.col("est_prop").sum().alias("est_sum_var1")
                ),
                on=vars_names[0],
                how="inner",
            )
            .join(
                other=tbl_df.group_by(vars_names[1]).agg(
                    pl.col("est_prop").sum().alias("est_sum_var2")
                ),
                on=vars_names[1],
                how="inner",
            )
            .with_columns(
                est_prop_null=pl.col("est_sum_var1")
                * pl.col("est_sum_var2")
                * pl.col("est_prop").sum()
            )
        )

        chisq_p = (
            df.shape[0]
            * (
                (tbl_df["est_prop"] - tbl_df["est_prop_null"]) ** 2
                / tbl_df["est_prop_null"]
            ).sum()
        )
        chisq_lr = (
            2
            * df.shape[0]
            * (
                tbl_df["est_prop"]
                * (tbl_df["est_prop"] / tbl_df["est_prop_null"]).log()
            )
            .fill_nan(0)
            .sum()
        )

        trace_delta = np.trace(delta_est)
        if trace_delta != 0:
            f_p = float(chisq_p / trace_delta)
            f_lr = float(chisq_lr / trace_delta)
            df_num = float((trace_delta ** 2) / np.trace(delta_est @ delta_est))
            df_den = float((tbl_est.nb_psus - tbl_est.nb_strata) * df_num)
        else:
            f_p = 0  # np.nan
            f_lr = 0  # np.nan
            df_num = 0  # np.nan
            df_den = 0  # np.nan

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

        self.row_levels = row_labels.to_list()
        self.col_levels = col_labels.to_list()
        self.vars_names = vars_names
        self.design_info = {
            "nb_strata": tbl_est.nb_strata,
            "nb_psus": tbl_est.nb_psus,
            "nb_obs": df.shape[0],
            "design_effect": 0,
            "degrees_of_freedom": tbl_est.nb_psus - tbl_est.nb_strata,
        }

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        both_levels = [self.row_levels, self.col_levels]
        twoway_df = pd.DataFrame([ll for ll in itertools.product(*both_levels)])
        twoway_df.columns = self.vars_names

        for _ in range(len(self.row_levels)):
            for _ in range(len(self.col_levels)):
                twoway_df[self.param] = sum(
                    pd.DataFrame.from_dict(
                        self.point_est, orient="index"
                    ).values.tolist(),
                    [],
                )
                twoway_df["stderror"] = sum(
                    pd.DataFrame.from_dict(
                        self.stderror, orient="index"
                    ).values.tolist(),
                    [],
                )
                twoway_df["lower_ci"] = sum(
                    pd.DataFrame.from_dict(
                        self.lower_ci, orient="index"
                    ).values.tolist(),
                    [],
                )
                twoway_df["upper_ci"] = sum(
                    pd.DataFrame.from_dict(
                        self.upper_ci, orient="index"
                    ).values.tolist(),
                    [],
                )
        # twoway_df.sort_values(by=self.vars_names, inplace=True)

        return twoway_df
