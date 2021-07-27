from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np

# import pandas as pd
import statsmodels.api as sm

from samplics.estimation.expansion import TaylorEstimator
from samplics.utils.types import Array, Number, Series, StringNumber
from samplics.utils.formats import fpc_as_dict, numpy_array, remove_nans, dict_to_dataframe


class SurveyGLM:
    """General linear models under complex survey sampling"""

    def __init__(self):
        pass

    @staticmethod
    def _residuals(e: np.ndarray, psu: np.ndarray, nb_vars: Number) -> tuple(np.ndarray, Number):

        psus, nh = np.unique(psu, return_counts=True)
        e_values = np.zeros(psus.shape[0], nb_vars)
        for i, p in enumerate(np.unique(psus.shape(0))):
            e_values[i, :] += np.sum(e[psu == p, :], axis=0)
        e_means = np.sum(e_values, axis=0) / nh

        return np.transpose(e_values - e_means) @ (e_values - e_means), nh

    def _calculate_g(
        self,
        samp_weight: np.ndarray,
        resid: np.ndarray,
        x: np.ndarray,
        stratum: Optional[np.ndarray],
        psu: Optional[np.ndarray],
    ) -> np.ndarray:

        e = samp_weight * resid * x

        if stratum is None:
            e_h, n_h = self._residuals(e=e, psu=psu, nb_vars=x.shape[1])
            return (n_h * (1 - self.fpc) / (n_h - 1)) * e_h
        else:
            g_h = np.zeros(x.shape)
            for s in np.unique(stratum):
                e_s = e[stratum == s, :]
                psu_s = psu[stratum == s]
                e_h, n_h = self._residuals(e=e_s, psu=psu_s, nb_vars=x.shape[1])
                g_h += (n_h * (1 - self.fpc[s]) / (n_h - 1)) * e_h
            return g_h

    def fit(
        self,
        y: Array,
        x: Optional[Array] = None,
        samp_weight: Optional[Array] = None,
        stratum: Optional[Series] = None,
        psu: Optional[Series] = None,
        fpc: Union[dict[StringNumber, Number], Series, Number] = 1.0,
        remove_nan: bool = False,
    ) -> None:

        if not isinstance(fpc, dict):
            self.fpc = fpc_as_dict(stratum, fpc)
        else:
            if list(np.unique(stratum)) != list(fpc.keys()):
                raise AssertionError("fpc dictionary keys must be the same as the strata!")
            else:
                self.fpc = fpc

        glm_model = sm.GLM(y=y, x=x, var_weights=samp_weight)
        glm_results = glm_model.fit()

        g = self._calculate_g(
            samp_weight=samp_weight, resid=glm_results.resid_response, x=x, psu=psu, fpc=fpc
        )

        d = glm_results.cov_params() / glm_results.scale

        self.beta = glm_results.params
        self.cov_beta = d * g * d
