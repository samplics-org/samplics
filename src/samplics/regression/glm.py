# from __future__ import annotations

import sys
import warnings

from typing import Any, Optional, Tuple, Union

import numpy as np
import polars as pl
import statsmodels.api as sm

from scipy import stats

from samplics.utils.basic_functions import get_single_psu_strata
from samplics.utils.checks import (
    _certainty_singleton,
    _combine_strata,
    _raise_singleton_error,
    _skip_singleton,
)
from samplics.utils.formats import fpc_as_dict, numpy_array
from samplics.utils.types import Array, ModelType, Number, Series, SinglePSUEst, StringNumber


class SurveyGLM:
    """General linear models under complex survey sampling"""

    def __init__(self, model: ModelType, alpha: float = 0.05):
        self.model = model
        self.alpha = alpha
        self.beta: dict = {}
        self.beta_cov: dict = {}
        self.x_labels: list[str] = []
        self.odds_ratio: dict = {}
        self.sample_size = 0
        self.single_psu_strata: Any = None

    def __str__(self) -> str:
        df = pl.DataFrame(
            {
                "label": self.x_labels,
                "coef": self.beta["point_est"],
                "stderr": self.beta["stderror"],
                "z": self.beta["z"],
                "p_value": self.beta["p_value"],
                "ci_low": self.beta["lower_ci"],
                "ci_upp": self.beta["upper_ci"],
            }
        )

        output = f"\n Model: {self.model.name} \n Sample size: {self.sample_size} \n Degree of freedom: {self.sample_size - len(self.beta['point_est'])} \n Alpha: {self.alpha} \n \n {df.__repr__()}"

        return output

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _residuals(e: np.ndarray, psu: np.ndarray, nb_vars: int) -> Tuple[np.ndarray, int]:
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
        x_cat_reference: Optional[dict[str, str]] = None,
        samp_weight: Optional[Array] = None,
        stratum: Optional[Series] = None,
        psu: Optional[Series] = None,
        fpc: Union[dict[str, Number], Series, Number] = 1.0,
        add_intercept: bool = False,
        tol: float = 1e-8,
        maxiter: int = 100,
        single_psu: Union[SinglePSUEst, dict[StringNumber, SinglePSUEst]] = SinglePSUEst.error,
        strata_comb: Optional[dict[Array, Array]] = None,
        remove_nan: bool = False,
    ) -> None:
        _y = numpy_array(y)
        _x = numpy_array(x) if x is not None else np.empty((len(_y), 0))
        _x_cat = numpy_array(x_cat) if x_cat is not None else np.empty(())
        _psu = numpy_array(psu) if psu is not None else np.empty(())
        _ssu = np.empty(())
        _stratum = numpy_array(stratum) if stratum is not None else np.empty(())
        _samp_weight = (
            numpy_array(samp_weight) if samp_weight is not None else np.ones(_y.shape[0])
        )

        if _x.ndim == 1 and _x.shape[0] > 0:
            _x = _x.reshape(-1, 1)

        if add_intercept:
            _x = np.insert(_x, 0, 1, axis=1)
            self.x_labels = ["Intercept"] + (
                x_labels if x_labels is not None else [f"__x{i}__" for i in range(1, _x.shape[1])]
            )
        else:
            self.x_labels = (
                x_labels if x_labels is not None else [f"__x{i}__" for i in range(_x.shape[1])]
            )

        if _samp_weight.shape in ((), (0,)):
            _samp_weight = np.ones(_y.shape[0])
        elif _samp_weight.shape[0] == 1:
            _samp_weight = _samp_weight * np.ones(_y.shape[0])

        if not isinstance(fpc, dict):
            self.fpc = fpc_as_dict(_stratum, fpc)
        else:
            if np.unique(_stratum).tolist() != list(fpc.keys()):
                raise AssertionError("fpc dictionary keys must be the same as the strata!")
            self.fpc = fpc

        df = pl.from_numpy(_x)
        df.columns = self.x_labels

        df = df.with_columns(pl.Series("_y", _y), pl.Series("_samp_weight", _samp_weight))

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
            for col in x_cat_labels:
                ref_val = None
                if x_cat_reference and col in x_cat_reference:
                    ref_val = x_cat_reference[col]
                levels = sorted(df[col].unique().to_list())
                if ref_val is not None:
                    levels = [ref_val] + [v for v in levels if v != ref_val]
                else:
                    ref_val = levels[0]
                for level in levels[1:]:
                    dummy_col = f"{col}_{level}".replace(".", "__")
                    df = df.with_columns(
                        pl.when(pl.col(col) == level).then(1.0).otherwise(0.0).alias(dummy_col)
                    )
                    self.x_labels.append(dummy_col)

        if _stratum.shape not in ((), (0,)):
            # TODO: we could improve efficiency by creating the pair [stratum,psu, ssu] ounce and
            # use it in get_single_psu_strata and in the uncertainty calculation functions
            self.single_psu_strata = get_single_psu_strata(_stratum, _psu)

        skipped_strata = np.array(None)
        if self.single_psu_strata is not None:
            if single_psu == SinglePSUEst.error:
                _raise_singleton_error(self.single_psu_strata)
            if single_psu == SinglePSUEst.skip:
                skipped_strata = _skip_singleton(
                    single_psu_strata=self.single_psu_strata, skipped_strata=self.single_psu_strata
                )
            if single_psu == SinglePSUEst.certainty:
                _psu = _certainty_singleton(
                    singletons=self.single_psu_strata,
                    _stratum=_stratum,
                    _psu=_psu,
                    _ssu=_ssu,
                )
                df = df.with_columns(pl.Series("_psu", np.asarray(_psu)))
            if single_psu == SinglePSUEst.combine:
                _stratum = _combine_strata(strata_comb, _stratum)
                df = df.with_columns(pl.Series("_stratum", np.asarray(_stratum)))
            # TODO: more method for singleton psus to be implemented
            if isinstance(single_psu, dict):
                for s in single_psu:
                    if single_psu[s] == SinglePSUEst.error:
                        _raise_singleton_error(self.single_psu_strata)
                    if single_psu[s] == SinglePSUEst.skip:
                        skipped_strata = _skip_singleton(
                            single_psu_strata=self.single_psu_strata, skipped_strata=numpy_array(s)
                        )
                        df = df.filter(~pl.col("_stratum").is_in(skipped_strata))
                        _stratum = df["_stratum"].to_numpy()
                        _psu = df["_psu"].to_numpy()
                    if single_psu[s] == SinglePSUEst.certainty:
                        _psu = _certainty_singleton(
                            singletons=numpy_array(s),
                            _stratum=_stratum,
                            _psu=_psu,
                            _ssu=_ssu,
                        )
                        df = df.with_columns(pl.Series("_psu", np.asarray(_psu)))
                        _stratum = df["_stratum"].to_numpy()
                        _psu = df["_psu"].to_numpy()
                    if single_psu[s] == SinglePSUEst.combine:
                        _stratum = _combine_strata(strata_comb, _stratum)
                        df = df.with_columns(pl.Series("_stratum", np.asarray(_stratum)))
                        _stratum = df["_stratum"].to_numpy()
                        _psu = df["_psu"].to_numpy()
                # df = df.with_columns(
                #     pl.Series("_stratum", np.asarray(_stratum)),
                #     pl.Series("_psu", np.asarray(_psu)),
                # )
            skipped_strata = get_single_psu_strata(_stratum, _psu)
            df = df.filter(~pl.col("_stratum").is_in(skipped_strata))
            if skipped_strata is not None and single_psu in [
                SinglePSUEst.certainty,
                SinglePSUEst.combine,
            ]:  # TODO: add the left our singletons when using the dict instead of SinglePSUEst
                _raise_singleton_error(self.single_psu_strata)

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

        self.sample_size = df.shape[0]
        self.beta_cov = (d @ g) @ d
        self.beta["point_est"] = glm_results.params
        self.beta["stderror"] = np.sqrt(np.diag(self.beta_cov))

        min_sd = 100 * sys.float_info.epsilon
        self.beta["z"] = np.zeros_like(self.beta["point_est"])
        for k, sd in enumerate(self.beta["stderror"]):
            if sd > min_sd:
                self.beta["z"][k] = self.beta["point_est"][k] / sd
            else:
                self.beta["z"][k] = self.beta["point_est"][k] / min_sd
                print(f"Warning: stderror is close to zero. stderror is smaller than {min_sd}")

        self.beta["p_value"] = 2 * stats.t.sf(
            np.abs(self.beta["z"]), self.sample_size - len(self.beta["point_est"])
        )
        crit = stats.norm.ppf(1 - self.alpha / 2)
        self.beta["lower_ci"] = self.beta["point_est"] - crit * self.beta["stderror"]
        self.beta["upper_ci"] = self.beta["point_est"] + crit * self.beta["stderror"]

        if self.model == ModelType.LOGISTIC:
            max_exp = sys.float_info.max / 100
            max_log = np.log(max_exp)
            self.odds_ratio["point_est"] = np.zeros_like(self.beta["point_est"])
            self.odds_ratio["lower_ci"] = np.zeros_like(self.beta["point_est"])
            self.odds_ratio["upper_ci"] = np.zeros_like(self.beta["point_est"])
            for k, beta in enumerate(self.beta["point_est"]):
                if beta < max_log:
                    self.odds_ratio["point_est"][k] = np.exp(self.beta["point_est"][k])
                    self.odds_ratio["lower_ci"][k] = np.exp(self.beta["lower_ci"][k])
                    self.odds_ratio["upper_ci"][k] = np.exp(self.beta["upper_ci"][k])
                else:
                    self.odds_ratio["point_est"][k] = max_exp
                    self.odds_ratio["lower_ci"][k] = max_exp
                    self.odds_ratio["upper_ci"][k] = max_exp
                    warnings.warn(
                        f"Exponentiation of beta values is potentially unsafe. Odds ratio capped to {max_exp}."
                    )
