from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from samplics.utils.formats import fpc_as_dict, numpy_array
from samplics.utils.types import Array, ModelType, Number, Series, StringNumber


class SurveyGLM:
    """General linear models under complex survey sampling"""

    def __init__(self, model: ModelType, alpha: float = 0.05):
        self.model = model
        self.alpha = alpha
        self.beta: dict = {}
        self.beta_cov: dict = {}
        self.x_labels: list[str] = []
        self.odds_ratio: dict = {}

    @staticmethod
    def _residuals(
        e: np.ndarray, psu: np.ndarray, nb_vars: Number
    ) -> tuple(np.ndarray, Number):
        psus = np.unique(psu)
        if psus.shape[0] == 1 and e.shape[0] == 1:
            raise AssertionError("Only one observation in the stratum")
        if psus.shape[0] == 1:
            psu = np.arange(e.shape[0])
            psus = np.unique(psu)
        e_values = np.zeros((psus.shape[0], nb_vars))

        for i, p in enumerate(np.unique(psus)):
            e_values[i, :] += np.sum(e[psu == p, :], axis=0)
        e_means = np.sum(e_values, axis=0) / psus.shape[0]

        return np.transpose(e_values - e_means) @ (e_values - e_means), psus.shape[0]

    def _calculate_g(
        self,
        samp_weight: np.ndarray,
        resid: np.ndarray,
        x: np.ndarray,
        stratum: np.ndarray,
        psu: np.ndarray,
        fpc: Union[dict[StringNumber, Number], Number],
        glm_scale=Number,
    ) -> np.ndarray:
        e = (samp_weight * resid)[:, None] * x / glm_scale
        if psu.shape in ((), (0,)):
            psu = np.arange(e.shape[0])
        if stratum.shape in ((), (0,)):
            e_h, n_h = self._residuals(e=e, psu=psu, nb_vars=x.shape[1])
            return fpc * (n_h / (n_h - 1)) * e_h
        else:
            g_h = np.zeros((x.shape[1], x.shape[1]))
            for s in np.unique(stratum):
                e_s = e[stratum == s, :]
                psu_s = psu[stratum == s]
                e_h, n_h = self._residuals(e=e_s, psu=psu_s, nb_vars=x.shape[1])
                g_h += fpc[s] * (n_h / (n_h - 1)) * e_h
            return g_h

    def estimate(
        self,
        y: Array,
        x: Optional[Array] = None,
        x_labels: Optional[list] = None,
        x_cat: Optional[Array] = None,
        x_cat_labels: Optional[list] = None,
        samp_weight: Optional[Array] = None,
        stratum: Optional[Series] = None,
        psu: Optional[Series] = None,
        fpc: Union[dict[StringNumber, Number], Series, Number] = 1.0,
        add_intercept: bool = False,
        tol: float = 1e-8,
        maxiter: int = 100,
        remove_nan: bool = False,
    ) -> None:
        _y = numpy_array(y)
        _x = numpy_array(x)
        _x_cat = numpy_array(x_cat)
        _psu = numpy_array(psu)
        _stratum = numpy_array(stratum)
        _samp_weight = numpy_array(samp_weight)

        if _x.shape[0] > 0 and _x.ndim == 1:
            _x = _x.reshape(_x.shape[0], 1)

        if add_intercept:
            _x = np.insert(_x, 0, 1, axis=1)
            self.x_labels = (
                ["intercept"] + [f"__x{i}__" for i in range(1, _x.shape[1])]
                if x_labels is None
                else ["intercept"] + x_labels
            )
        else:
            self.x_labels = (
                ["intercept"] + [f"__x{i}__" for i in range(1, _x.shape[1])]
                if x_labels is None
                else x_labels
            )

        if _samp_weight.shape in ((), (0,)):
            _samp_weight = np.ones(y.shape[0])
        if _samp_weight.shape[0] == 1:
            _samp_weight = _samp_weight * np.ones(_y.shape[0])

        if not isinstance(fpc, dict):
            self.fpc = fpc_as_dict(_stratum, fpc)
        else:
            if np.unique(_stratum).tolist() != list(fpc.keys()):
                raise AssertionError(
                    "fpc dictionary keys must be the same as the strata!"
                )
            else:
                self.fpc = fpc

        df = pl.from_numpy(_x)
        df.columns = self.x_labels
        df = df.with_columns(
            pl.Series("_y", _y),
            pl.Series("_samp_weight", _samp_weight),
        )

        if _x_cat.shape != ():
            df_cat = pl.from_numpy(_x_cat)
            df_cat.columns = x_cat_labels
            df = df.hstack(df_cat)

        if _stratum.shape != ():
            df = df.with_columns(pl.Series("_stratum", _stratum))

        if _psu.shape != ():
            df = df.with_columns(pl.Series("_psu", _psu))

        if remove_nan:
            df = df.drop_nulls().drop_nans()

        if _x_cat.shape != ():
            _x_cat = df_cat.select(x_cat_labels).unique().sort(x_cat_labels)
            for col in x_cat_labels:
                col_values = _x_cat[col].unique()
                col_dummies = col_values.to_dummies(drop_first=True)
                col_df = col_dummies.with_columns(
                    pl.fold(
                        acc=pl.lit(0.0),
                        function=lambda acc, x: acc + x,
                        exprs=[pl.col(c) for c in col_dummies.columns],
                    ).alias("sum")
                )
                col_df = col_df.with_columns(
                    [
                        pl.when(pl.col("sum") == 0)
                        .then(pl.lit(-1.0))
                        .otherwise(pl.col(col))
                        .alias(col)
                        for col in col_dummies.columns
                    ]
                ).insert_column(0, col_values)
                self.x_labels = self.x_labels + col_dummies.columns
                df = df.join(col_df.drop("sum"), on=col, how="left")

        match self.model:
            case ModelType.LINEAR:
                glm_model = sm.GLM(
                    endog=df["_y"].to_numpy(),
                    exog=df.select(self.x_labels).to_numpy(),
                    var_weights=df["_samp_weight"].to_numpy(),
                )
            case ModelType.LOGISTIC:
                glm_model = sm.GLM(
                    endog=df["_y"].to_numpy(),
                    exog=df.select(self.x_labels).to_numpy(),
                    var_weights=df["_samp_weight"].to_numpy(),
                    family=sm.families.Binomial(),
                )
            case _:
                raise NotImplementedError(f"Model {self.model} is not implemented yet")

        glm_results = glm_model.fit()
        g = self._calculate_g(
            samp_weight=df["_samp_weight"].to_numpy(),
            resid=glm_results.resid_response,
            x=df.select(self.x_labels).to_numpy(),
            stratum=df["_stratum"].to_numpy() if _stratum.shape != () else _stratum,
            psu=df["_psu"].to_numpy() if _psu.shape != () else _psu,
            fpc=self.fpc,
            glm_scale=glm_results.scale,
        )
        d = glm_results.cov_params()

        self.beta_cov = (d @ g) @ d
        self.beta["point_est"] = glm_results.params
        self.beta["stderror"] = np.sqrt(np.diag(self.beta_cov))
        self.beta["z"] = self.beta["point_est"] / self.beta["stderror"]
        self.beta["p_value"] = stats.norm.sf(np.abs(self.beta["z"])) * 2  # sf = 1 - cdf
        self.beta["lower_ci"] = (
            self.beta["point_est"]
            - stats.norm.ppf(1 - self.alpha / 2) * self.beta["stderror"]
        )
        self.beta["upper_ci"] = (
            self.beta["point_est"]
            + stats.norm.ppf(1 - self.alpha / 2) * self.beta["stderror"]
        )

        self.odds_ratio["point_est"] = np.exp(self.beta["point_est"])
        self.odds_ratio["lower_ci"] = np.exp(self.beta["lower_ci"])
        self.odds_ratio["upper_ci"] = np.exp(self.beta["upper_ci"])
