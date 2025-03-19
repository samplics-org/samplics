from __future__ import annotations

from typing import Optional, Union, Tuple

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
        e: np.ndarray, psu: np.ndarray, nb_vars: int
    ) -> Tuple[np.ndarray, int]:
        unique_psus = np.unique(psu)

        if unique_psus.shape[0] == 1 and e.shape[0] == 1:
            raise AssertionError("Only one observation in the stratum")

        if unique_psus.shape[0] == 1:
            psu = np.arange(e.shape[0])
            unique_psus = np.unique(psu)

        e_values = np.vstack([np.sum(e[psu == p, :], axis=0) for p in unique_psus])
        e_means = e_values.mean(axis=0)
        ss = (e_values - e_means).T @ (e_values - e_means)

        return ss, unique_psus.size

    def _calculate_g(
        self,
        samp_weight: np.ndarray,
        resid: np.ndarray,
        x: np.ndarray,
        stratum: Optional[np.ndarray],
        psu: Optional[np.ndarray],
        fpc: Union[dict[str, Number], Number],
        glm_scale: Number,
    ) -> np.ndarray:
        e = (samp_weight * resid)[:, None] * x / glm_scale

        if psu.shape in ((), (0,)):
            psu = np.arange(e.shape[0])

        if stratum.shape in ((), (0,)):
            e_h, n_h = self._residuals(e=e, psu=psu, nb_vars=x.shape[1])
            return fpc * (n_h / (n_h - 1)) * e_h
        else:
            unique_strata = np.unique(stratum)
            g_h = np.zeros((x.shape[1], x.shape[1]))
            for s in unique_strata:
                mask = stratum == s
                e_s = e[mask, :]
                psu_s = psu[mask]
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
        fpc: Union[dict[str, Number], Series, Number] = 1.0,
        add_intercept: bool = False,
        tol: float = 1e-8,
        maxiter: int = 100,
        remove_nan: bool = False,
    ) -> None:
        # Convert inputs to numpy arrays for consistency.
        _y = numpy_array(y)
        _x = numpy_array(x) if x is not None else np.empty((len(_y), 0))
        _x_cat = numpy_array(x_cat) if x_cat is not None else np.empty(())
        _psu = numpy_array(psu) if psu is not None else np.empty(())
        _stratum = numpy_array(stratum) if stratum is not None else np.empty(())
        _samp_weight = (
            numpy_array(samp_weight)
            if samp_weight is not None
            else np.ones(_y.shape[0])
        )

        # Ensure _x is 2-dimensional.
        if _x.ndim == 1 and _x.shape[0] > 0:
            _x = _x.reshape(-1, 1)

        # Add intercept if requested.
        if add_intercept:
            _x = np.insert(_x, 0, 1, axis=1)
            self.x_labels = ["intercept"] + (
                x_labels
                if x_labels is not None
                else [f"__x{i}__" for i in range(1, _x.shape[1])]
            )
        else:
            self.x_labels = (
                x_labels
                if x_labels is not None
                else [f"__x{i}__" for i in range(_x.shape[1])]
            )

        # Ensure samp_weight is of proper length.
        if _samp_weight.shape in ((), (0,)):
            _samp_weight = np.ones(_y.shape[0])
        elif _samp_weight.shape[0] == 1:
            _samp_weight = _samp_weight * np.ones(_y.shape[0])

        # Handle finite population correction: convert fpc to dictionary if not already.
        if not isinstance(fpc, dict):
            self.fpc = fpc_as_dict(_stratum, fpc)
        else:
            if np.unique(_stratum).tolist() != list(fpc.keys()):
                raise AssertionError(
                    "fpc dictionary keys must be the same as the strata!"
                )
            self.fpc = fpc

        # Create Polars DataFrame from _x and assign column labels.
        df = pl.from_numpy(_x)
        df.columns = self.x_labels

        # Add response and weight columns.
        df = df.with_columns(
            pl.Series("_y", _y), pl.Series("_samp_weight", _samp_weight)
        )

        # Optionally add categorical predictors.
        if _x_cat.shape != ():
            df_cat = pl.from_numpy(_x_cat)
            df_cat.columns = x_cat_labels
            df = df.hstack(df_cat)

        # Add survey design variables if provided.
        if _stratum.shape != ():
            df = df.with_columns(pl.Series("_stratum", _stratum))
        if _psu.shape != ():
            df = df.with_columns(pl.Series("_psu", _psu))

        # Remove rows with null or NaN values if specified.
        if remove_nan:
            df = df.drop_nulls().drop_nans()

        # Process categorical variables (if provided) to create dummy variables.
        if _x_cat.shape != ():
            # Get unique combinations for the categorical columns and sort.
            unique_cat = df.select(x_cat_labels).unique().sort(x_cat_labels)
            for col in x_cat_labels:
                # Create dummy variables from unique values.
                col_values = unique_cat[col].unique()
                col_dummies = col_values.to_dummies(drop_first=True)
                col_dummies.columns = [
                    col.replace(".", "__") for col in col_dummies.columns
                ]
                # Sum across dummy columns to create an indicator.
                # col_df = col_dummies.with_columns(
                #     pl.fold(
                #         acc=pl.lit(0.0),
                #         function=lambda acc, x: acc + x,
                #         exprs=[pl.col(c) for c in col_dummies.columns]
                #     ).alias("sum")
                # )
                # # Replace dummy values if the sum is zero.
                # col_df = col_df.with_columns(
                #     [
                #         pl.when(pl.col("sum") == 0)
                #         .then(pl.lit(0.0))
                #         .otherwise(pl.col(c))
                #         .alias(c)
                #         for c in col_dummies.columns
                #     ]
                # ).insert_column(0, col_values)
                # df = df.join(col_df.drop("sum"), on=col, how="left")
                self.x_labels += col_dummies.columns
                df = df.join(
                    col_dummies.insert_column(0, col_values), on=col, how="left"
                )

        # Fit model based on specified model type.
        match self.model:
            case ModelType.LINEAR:
                glm_model = sm.GLM(
                    endog=df["_y"].to_numpy(),
                    exog=df.select(self.x_labels).to_numpy(),
                    freq_weights=df["_samp_weight"].to_numpy(),
                    family=sm.families.Gaussian(),
                )
            case ModelType.LOGISTIC:
                glm_model = sm.GLM(
                    endog=df["_y"].to_numpy(),
                    exog=df.select(self.x_labels).to_numpy(),
                    freq_weights=df["_samp_weight"].to_numpy(),
                    family=sm.families.Binomial(),
                )
                # import statsmodels.formula.api as smf
                # glm_model = smf.glm(
                #     formula="_y ~ " + " + ".join([f"{col}" for col in self.x_labels if col != "intercept"]),
                #     family=sm.families.Binomial(),
                #     freq_weights=df["_samp_weight"].to_numpy(),
                #     data=df,
                # )
                # breakpoint()
            case _:
                raise NotImplementedError(f"Model {self.model} is not implemented yet")

        # Continue with fitting, iteration, or returning the fitted model as needed.

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
