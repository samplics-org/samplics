"""Replicate weights module.

The module has one main class called *ReplicateWeight* which implements three replication 
techniques: Bootstrap, Jackknife, and balanced repeated replication (BRR). For reference, 
users can consultant Efron, B. and Tibshirani, R.J. (1994) [#et1994]_, Valliant, R. and 
Dever, J. A. (2018) [#vd2018]_ and Wolter, K.M. (2007) [#w2007]_ for more details. 

.. [#et1994] Efron, B. and Tibshirani, R.J. (1994), *An Introduction to the Boostrap*, 
   Chapman & Hall/CRC.
.. [#w2007] Wolter, K.M. (2007), *Introduction to Variance Estimate, 2nd edn.*, 
   Springer-Verlag New York, Inc
"""

from __future__ import annotations

import math

from typing import Optional, Union

import numpy as np
import pandas as pd

from samplics.utils import checks, formats
from samplics.utils import hadamard as hdd
from samplics.utils.types import Array, Number


class ReplicateWeight:
    """*ReplicateWeight* implements Boostrap, Jackknife and BRR to derive replicate weights.
    When possible design weights should be used as the input weights for creating the replicate
    weights, hence the weight adjustments can be applied to the replicates.

    Attributes:
        | method (str): replicate method.
        | fay_coef (float): Fay coefficient when implementing BRR-Fay.
        | number_reps (int): number of replicates.
        | rep_coefs (np.ndarray): coefficients associated to the replicates.
        | stratification (bool): stratification indicator.
        | number_psus (int): number of primary sampling units.
        | number_strata (int): number of strata.
        | random_seed (int): random seed.


    Methods:
        | replicate(): computes the replicate weights.

    """

    def __init__(
        self,
        method: str,
        stratification: bool = True,
        number_reps: int = 500,
        fay_coef: float = 0.0,
        random_seed: Optional[int] = None,
    ):

        self.method = method.lower()
        self.stratification = stratification
        if self.method == "bootstrap":
            self.number_reps = number_reps
            self.rep_coefs = list((1 / number_reps) * np.ones(number_reps))
        elif self.method == "brr":
            self.number_reps = 0
            self.fay_coef = fay_coef

        self.number_psus = 0
        self.number_strata = 0
        self.rep_coefs = []
        self.degree_of_freedom = 0
        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(random_seed)

    def _reps_to_dataframe(
        self, psus: pd.DataFrame, rep_data: np.ndarray, rep_prefix: str
    ) -> pd.DataFrame:

        rep_data = pd.DataFrame(rep_data)
        rep_data.reset_index(drop=True, inplace=True)
        rep_data.rename(columns=lambda x: rep_prefix + str(x + 1), inplace=True)
        psus.reset_index(drop=True, inplace=True)
        rep_data = pd.concat([psus, rep_data], axis=1)

        return rep_data

    def _rep_prefix(self, prefix: Optional[str]) -> str:

        if self.method == "jackknife" and prefix is None:
            rep_prefix = "_jk_wgt_"
        elif self.method == "bootstrap" and prefix is None:
            rep_prefix = "_boot_wgt_"
        elif self.method == "brr" and prefix is None:
            rep_prefix = "_brr_wgt_"
        elif self.method == "brr" and self.fay_coef > 0 and prefix is None:
            rep_prefix = "_fay_wgt_"
        elif prefix is None:
            rep_prefix = "_rep_wgt_"
        else:
            rep_prefix = prefix

        return rep_prefix

    def _degree_of_freedom(
        self,
        weight: np.ndarray,
        stratum: Optional[np.ndarray],
        psu: np.ndarray,
    ) -> None:

        stratum = formats.numpy_array(stratum)
        psu = formats.numpy_array(psu)

        if stratum.size <= 1:
            self.degree_of_freedom = np.unique(psu).size - 1
        elif psu.size > 1:
            self.degree_of_freedom = np.unique(psu).size - np.unique(stratum).size
        else:
            weight = formats.numpy_array(weight)
            self.degree_of_freedom = weight.size

    # Bootstrap methods
    @staticmethod
    def _boot_psus_replicates(
        number_psus: int,
        number_reps: int,
        samp_rate: Number = 0,
        size_gap: int = 1,
    ) -> np.ndarray:
        """Creates the bootstrap replicates structure"""

        if number_psus <= size_gap:
            raise AssertionError("size_gap should be smaller than the number of units")

        sample_size = number_psus - size_gap
        psu = np.arange(0, number_psus)
        psu_boot = np.random.choice(psu, size=(number_reps, sample_size))
        psu_replicates = np.zeros(shape=(number_psus, number_reps))
        for rep in np.arange(0, number_reps):
            psu_ids, psus_counts = np.unique(psu_boot[rep, :], return_counts=True)
            psu_replicates[:, rep][psu_ids] = psus_counts

        ratio_sqrt = np.sqrt((1 - samp_rate) * sample_size / (number_psus - 1))

        return np.asarray(
            1 - ratio_sqrt + ratio_sqrt * (number_psus / sample_size) * psu_replicates
        )

    def _boot_replicates(
        self,
        psu: np.ndarray,
        stratum: Optional[np.ndarray],
        samp_rate: Number = 0,
        size_gap: int = 1,
    ) -> np.ndarray:

        if stratum is None:
            psu_ids = np.unique(psu)
            boot_coefs = self._boot_psus_replicates(
                psu_ids.size, self.number_reps, samp_rate, size_gap
            )
        else:
            strata = np.unique(stratum)
            for k, s in enumerate(strata):
                psu_ids_s = np.unique(psu[stratum == s])
                number_psus_s = psu_ids_s.size
                boot_coefs_s = self._boot_psus_replicates(
                    number_psus_s, self.number_reps, samp_rate, size_gap
                )
                if k == 0:
                    boot_coefs = boot_coefs_s
                else:
                    boot_coefs = np.concatenate((boot_coefs, boot_coefs_s), axis=0)

        return boot_coefs

    # BRR methods
    def _brr_number_reps(self, psu: np.ndarray, stratum: Optional[np.ndarray] = None) -> None:

        if stratum is None:
            self.number_psus = np.unique(psu).size
            self.number_strata = self.number_psus // 2 + self.number_psus % 2
        else:
            self.number_psus = np.unique(np.array(list(zip(stratum, psu))), axis=0).shape[0]
            self.number_strata = np.unique(stratum).size
            if 2 * self.number_strata != self.number_psus:
                raise AssertionError("Number of psus must be twice the number of strata!")

        if self.number_reps < self.number_strata:
            self.number_reps = self.number_strata

        if self.number_reps <= 28:
            if self.number_reps % 4 != 0:
                self.number_reps = 4 * (self.number_reps // 4 + 1)
        else:
            nb_reps_log2 = int(math.log(self.number_reps, 2))
            if math.pow(2, nb_reps_log2) != self.number_reps:
                self.number_reps = int(math.pow(2, nb_reps_log2))

    def _brr_replicates(self, psu: np.ndarray, stratum: Optional[np.ndarray]) -> np.ndarray:
        """Creates the brr replicate structure"""

        if not (0 <= self.fay_coef < 1):
            raise ValueError("The Fay coefficient must be greater or equal to 0 and lower than 1.")
        self._brr_number_reps(psu, stratum)

        self.rep_coefs = list(
            (1 / (self.number_reps * pow(1 - self.fay_coef, 2))) * np.ones(self.number_reps)
        )

        brr_coefs = hdd.hadamard(self.number_reps).astype(float)
        brr_coefs = brr_coefs[:, 1 : self.number_strata + 1]
        brr_coefs = np.repeat(brr_coefs, 2, axis=1)
        for r in np.arange(self.number_reps):
            for h in np.arange(self.number_strata):
                start = 2 * h
                end = start + 2
                if brr_coefs[r, start] == 1.0:
                    brr_coefs[r, start:end] = [
                        self.fay_coef,
                        2 - self.fay_coef,
                    ]
                else:  # brr_coefs[r, 2 * h] == -1:
                    brr_coefs[r, start:end] = [
                        2 - self.fay_coef,
                        self.fay_coef,
                    ]

        return brr_coefs.T

    # Jackknife
    @staticmethod
    def _jkn_psus_replicates(number_psus: int) -> np.ndarray:
        """Creates the jackknife delete-1 replicate structure"""

        jk_coefs = (number_psus / (number_psus - 1)) * (
            np.ones((number_psus, number_psus)) - np.identity(number_psus)
        )

        return np.asarray(jk_coefs)

    def _jkn_replicates(self, psu: np.ndarray, stratum: Optional[np.ndarray]) -> np.ndarray:

        self.rep_coefs = ((self.number_reps - 1) / self.number_reps) * np.ones(self.number_reps)

        if stratum is None:
            psu_ids = np.unique(psu)
            jk_coefs = self._jkn_psus_replicates(psu_ids.size)
        else:
            strata = np.unique(stratum)
            jk_coefs = np.ones((self.number_reps, self.number_reps))
            start = end = 0
            for s in strata:
                psu_ids_s = np.unique(psu[stratum == s])
                number_psus_s = psu_ids_s.size
                end = start + number_psus_s
                jk_coefs[start:end, start:end] = self._jkn_psus_replicates(number_psus_s)
                self.rep_coefs[start:end] = (number_psus_s - 1) / number_psus_s
                start = end

        self.rep_coefs = list(self.rep_coefs)

        return jk_coefs

    def replicate(
        self,
        samp_weight: Array,
        psu: Array,
        stratum: Optional[Array] = None,
        rep_coefs: Union[Array, Number] = False,
        rep_prefix: Optional[str] = None,
        psu_varname: str = "_psu",
        str_varname: str = "_stratum",
    ) -> pd.DataFrame:
        """Computes replicate sample weights.

        Args:
            samp_weight (Array): array of sample weights. To incorporate the weights adjustment
                in the replicate weights, first replicate the design sample weights then apply
                the adjustments to the replicates.
            psu (Array):
            stratum (Array, optional): array of the strata. Defaults to None.
            rep_coefs (Union[Array, Number], optional): coefficients associated to the replicates.
                Defaults to False.
            rep_prefix (str, optional): prefix to apply to the replicate weights names.
                Defaults to None.
            psu_varname (str, optional): name of the psu variable in the output dataframe.
                Defaults to "_psu".
            str_varname (str, optional): name of the stratum variable in the output dataframe.
                Defaults to "_stratum".

        Raises:
            AssertionError: raises an assertion error when stratum is None for a stratified design.
            AssertionError: raises an assertion error when the replication method is not valid.

        Returns:
            pd.DataFrame: a dataframe of the replicates sample weights.
        """

        samp_weight = formats.numpy_array(samp_weight)
        psu = formats.numpy_array(psu)
        if not self.stratification:
            stratum = None
        else:
            stratum = formats.numpy_array(stratum)

        self._degree_of_freedom(samp_weight, stratum, psu)

        if self.stratification and stratum is None:
            raise AssertionError("For a stratified design, stratum must be specified.")
        elif stratum is not None:
            stratum_psu = pd.DataFrame({str_varname: stratum, psu_varname: psu})
            stratum_psu.sort_values(by=str_varname, inplace=True)
            key = [str_varname, psu_varname]
        elif self.method == "brr":
            _, str_index = np.unique(psu, return_index=True)
            checks.assert_brr_number_psus(str_index)
            psus = psu[np.sort(str_index)]
            strata = np.repeat(range(1, psus.size // 2 + 1), 2)
            stratum_psu = pd.DataFrame({str_varname: strata, psu_varname: psus})
            psu_pd = pd.DataFrame({psu_varname: psu})
            stratum_psu = pd.merge(psu_pd, stratum_psu, on=psu_varname, how="left", sort=False)
            stratum_psu = stratum_psu[[str_varname, psu_varname]]
            key = [str_varname, psu_varname]
        else:
            stratum_psu = pd.DataFrame({psu_varname: psu})
            key = [psu_varname]

        psus_ids = stratum_psu.drop_duplicates()

        if self.method == "jackknife":
            self.number_reps = psus_ids.shape[0]
            _rep_data = self._jkn_replicates(psu, stratum)
        elif self.method == "bootstrap":
            _rep_data = self._boot_replicates(psu, stratum)
        elif self.method == "brr":
            _rep_data = self._brr_replicates(psu, stratum)
            self.rep_coefs = list(
                (1 / self.number_reps * pow(1 - self.fay_coef, 2)) * np.ones(self.number_reps)
            )
        else:
            raise AssertionError(
                "Replication method not recognized. Possible options are: 'bootstrap', 'brr', and 'jackknife'"
            )

        rep_prefix = self._rep_prefix(rep_prefix)
        _rep_data = self._reps_to_dataframe(psus_ids, _rep_data, rep_prefix)

        samp_weight = pd.DataFrame({"_samp_weight": samp_weight})
        samp_weight.reset_index(drop=True, inplace=True)
        full_sample = pd.concat([stratum_psu, samp_weight], axis=1)
        full_sample = pd.merge(full_sample, _rep_data, on=key, how="left", sort=False)

        if not rep_coefs:
            rep_cols = [col for col in full_sample if col.startswith(rep_prefix)]
            full_sample[rep_cols] = full_sample[rep_cols].mul(samp_weight.values, axis=0)

        return full_sample

    def normalize(self, rep_weights: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        pass

    def adjust(self, rep_weights: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        pass

    def trim(self, rep_weights: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        pass
