from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np

# import pandas as pd
import statsmodels.api as sm

from samplics.estimation.expansion import TaylorEstimator
from samplics.utils.formats import dict_to_dataframe, fpc_as_dict, numpy_array, remove_nans
from samplics.utils.types import Array, Number, Series, StringNumber


class SurveyGLM:
    """General linear models under complex survey sampling"""

    def __init__(self):
        self.beta: np.ndarray

    @staticmethod
    def _residuals(e: np.ndarray, psu: np.ndarray, nb_vars: Number) -> tuple(np.ndarray, Number):

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
        stratum: Optional[np.ndarray],
        psu: Optional[np.ndarray],
        fpc: Union[dict[StringNumber, Number], Number],
        glm_scale=Number,
    ) -> np.ndarray:

        e = (samp_weight * resid)[:, None] * x / glm_scale
        if psu is None:
            psu = np.arange(e.shape[0])
        if stratum is None:
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
        samp_weight: Optional[Array] = None,
        stratum: Optional[Series] = None,
        psu: Optional[Series] = None,
        fpc: Union[dict[StringNumber, Number], Series, Number] = 1.0,
        remove_nan: bool = False,
    ) -> None:

        y = numpy_array(y)
        y_temp = y.copy()

        x = numpy_array(x) if x is not None else None
        psu = numpy_array(psu) if psu is not None else None

        if samp_weight is None:
            weight_temp = np.ones(y.shape[0])
        elif isinstance(samp_weight, (float, int)):
            weight_temp = samp_weight * np.ones(y_temp.shape[0])
        elif isinstance(samp_weight, np.ndarray):
            weight_temp = samp_weight.copy()
        else:
            weight_temp = np.asarray(samp_weight)

        if not isinstance(fpc, dict):
            self.fpc = fpc_as_dict(stratum, fpc)
        else:
            if list(np.unique(stratum)) != list(fpc.keys()):
                raise AssertionError("fpc dictionary keys must be the same as the strata!")
            else:
                self.fpc = fpc

        glm_model = sm.GLM(endog=y_temp, exog=x, var_weights=weight_temp)
        glm_results = glm_model.fit()

        g = self._calculate_g(
            samp_weight=samp_weight,
            resid=glm_results.resid_response,
            x=x,
            stratum=stratum,
            psu=psu,
            fpc=self.fpc,
            glm_scale=glm_results.scale,
        )

        d = glm_results.cov_params()

        self.beta = glm_results.params
        self.cov_beta = (d @ g) @ d
