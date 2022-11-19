"""Sampling selection module

The module has one main class called *SampleSelection* which provides a number of random selection
methods and associated probability of selection. All the samping techniques implemented in this
modules are discussed in the following reference book: Cochran, W.G. (1977) [#c1977]_, 
Kish, L. (1965) [#k1965]_, and Lohr, S.L. (2010) [#l2010]_. Furthermore, Brewer, K.R.W. and 
Hanif, M. (1983) [#bh1983]_ provides comprehensive and detailed descriptions of these complex 
sampling algorithms.

.. [#c1977] Cochran, W.G. (1977), *Sampling Techniques, 3rd edn.*, Jonh Wiley & Sons, Inc.
.. [#k1965] Kish, L. (1965), *Survey Sampling*, Jonh Wiley & Sons, Inc.
.. [#l2010] Lohr, S.L. (2010), *Sampling: Design and Analysis, 2nd edn.*, Cengage Learning, Inc.
.. [#bh1983] Brewer, K.R.W. and Hanif, M. (1983), *Sampling With Unequal Probabilities*,  
   Springer-Verlag New York, Inc
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from samplics.utils import formats
from samplics.utils.types import (
    Array,
    DictStrFloat,
    DictStrInt,
    DictStrNum,
    Number,
    SelectMethod,
)


@dataclass
class SampleSelection:
    """*SampleSelection* implements a number of sampling selection algorithms.

    The implemented sampling algorithms are the simple random sampling (srs), the
    systematic selection (sys), five different algorithms of probability proportional
    to size (pps), and the generic selection.

    Attributes of the class:
        | method (SelectMethod): a string specifying the sampling algorithm. The available
        |   sampling algorithms are: simple random sampling (srs), systematic selection (sys),
        |   Brewer's procedure (pps-brewer), Hanurav-Vijayan procedure (pps-hv),
        |   Murphy's procedure (pps-murphy), Rao-Sampford procedure (pps-rs), systematic
        |   pps (pps-sys), and a generic selection from arbitrary probabilities using
        |   *numpy.random.choice()* method.
        | strat (bool): indicates if the sampling procedure is stratified or not.
        | wr (bool): indicates whether the selection is with replacement.

    Main functions:
        | inclusion_probs(): provides the inclusion probabilities.
        | joint_inclusion_probs(): provides the joint probalities of selection.
        | select(): indicates the selected sample.

    TODO: handling of certainties and implementation of joint_inclusion_probs().
    """

    def __init__(
        self,
        method: SelectMethod,
        strat: bool = False,
        wr: bool = True,
    ) -> None:
        self.method: SelectMethod = method
        self.strat: bool = True if strat else False
        self.wr: bool = wr
        self.fpc: Union[DictStrNum, Number] = {}

    @staticmethod
    def _to_dataframe(
        samp_unit: np.ndarray,
        stratum: Optional[np.ndarray],
        mos: Optional[np.ndarray],
        sample: np.ndarray,
        hits: np.ndarray,
        probs: Optional[np.ndarray],
    ) -> pd.DataFrame:

        df = pd.DataFrame(
            {
                "_samp_unit": samp_unit,
                "_stratum": stratum,
                "_mos": mos,
                "_sample": sample,
                "_hits": hits,
                "_probs": probs,
            }
        )
        df.reset_index(drop=True, inplace=True)

        if stratum is None:
            df.drop(columns=["_stratum"], inplace=True)
        if mos is None:
            df.drop(columns=["_mos"], inplace=True)

        return df

    def _calculate_fpc(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
    ) -> None:

        samp_unit = formats.sample_units(samp_unit, unique=True)
        samp_size = formats.sample_size_dict(samp_size, self.strat, stratum)

        if isinstance(samp_size, dict):
            self.fpc = dict()
            strata = np.unique(stratum)
            for k, s in enumerate(strata):
                number_units_s = len(samp_unit[stratum == s])
                self.fpc[s] = np.sqrt((number_units_s - samp_size[s]) / (number_units_s - 1))
        else:
            self.fpc = np.sqrt((samp_unit.size - samp_size) / (samp_unit.size - 1))

    def _grs_select(
        self,
        probs: np.ndarray,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        samp_size = formats.sample_size_dict(samp_size, self.strat, stratum)
        sample = np.zeros(samp_unit.size).astype(bool)
        hits = np.zeros(samp_unit.size).astype(int)
        self._calculate_fpc(samp_unit, samp_size, stratum)
        if isinstance(samp_size, dict):
            all_indices = np.array(range(samp_unit.size))
            sampled_indices_list = []
            for s in np.unique(stratum):
                stratum_units = stratum == s
                probs_s = probs[stratum_units] / np.sum(probs[stratum_units])
                sampled_indices_s = np.random.choice(
                    all_indices[stratum_units],
                    samp_size[s],
                    self.wr,
                    probs_s,
                )
                sampled_indices_list.append(sampled_indices_s)
            sampled_indices = np.array(
                [val for sublist in sampled_indices_list for val in sublist]
            ).flatten()
        else:
            sampled_indices = np.random.choice(
                samp_unit.size,
                samp_size,
                self.wr,
                probs / np.sum(probs),
            )

        indices_s, hits_s = np.unique(sampled_indices, return_counts=True)
        sample[indices_s] = True
        hits[indices_s] = hits_s

        return sample, hits

    @staticmethod
    def _anycertainty(
        samp_size: Union[DictStrInt, int],
        stratum: Optional[np.ndarray],
        mos: np.ndarray,
    ) -> bool:

        if stratum is not None and isinstance(samp_size, dict):
            probs = np.zeros(stratum.size)
            for s in np.unique(stratum):
                stratum_units = stratum == s
                mos_s = mos[stratum_units]
                probs[stratum_units] = samp_size[s] * mos_s / np.sum(mos_s)
        elif isinstance(samp_size, (int, float)):
            probs = samp_size * mos / np.sum(mos)

        return bool((probs >= 1).any())  # to make mypy happy

    # SRS methods
    def _srs_inclusion_probs(
        self,
        samp_unit: Array,
        samp_size: Union[DictStrInt, int],
        stratum: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        samp_unit = formats.sample_units(samp_unit)
        samp_size = formats.sample_size_dict(samp_size, self.strat, stratum)

        number_units = samp_unit.size
        incl_probs: np.ndarray
        if stratum is not None and isinstance(samp_size, dict):
            incl_probs = np.zeros(number_units) * np.nan
            for s in np.unique(stratum):
                number_units_s = samp_unit[stratum == s].size
                incl_probs[stratum == s] = samp_size[s] / number_units_s
        elif isinstance(samp_size, (int, float)):
            number_units = samp_unit.size
            incl_probs = np.ones(number_units) * samp_size / number_units
        else:
            raise TypeError("samp_size has the wrong type!")

        return incl_probs

    # PPS methods
    def _pps_inclusion_probs(
        self,
        samp_unit: Array,
        samp_size: Union[DictStrInt, int],
        mos: np.ndarray,
        stratum: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        samp_unit = formats.sample_units(samp_unit, unique=True)
        samp_size = formats.sample_size_dict(samp_size, self.strat, stratum)

        incl_probs: np.ndarray
        if isinstance(samp_size, dict):
            number_units = samp_unit.size
            incl_probs = np.zeros(number_units) * np.nan
            for s in np.unique(stratum):
                stratum_units = stratum == s
                mos_s = mos[stratum_units]
                incl_probs[stratum_units] = samp_size[s] * mos_s / np.sum(mos_s)
        else:
            incl_probs = samp_size * mos / np.sum(mos)

        return incl_probs

    @staticmethod
    def _pps_sys_select(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        cumsize = np.append(0, np.cumsum(mos))
        samp_interval = cumsize[-1] / samp_size
        random_start = np.random.random_sample() * samp_interval
        random_picks = random_start + samp_interval * np.linspace(0, samp_size - 1, samp_size)

        hits = np.zeros(samp_unit.size).astype(int)
        for k in range(cumsize.size - 1):
            for ll in range(random_picks.size):
                if cumsize[k] < random_picks[ll] <= cumsize[k + 1]:
                    hits[k] += 1

        return np.asarray(hits >= 1), hits

    @staticmethod
    def _pps_hv_select(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        pop_size = samp_unit.size
        all_indices = np.arange(pop_size)
        size_order = mos.argsort()
        all_indices_sorted = all_indices[size_order]
        mos_sorted = mos[size_order]
        probs_sorted = np.append(mos_sorted / np.sum(mos_sorted), 1 / samp_size)
        last_nplus1_probs = probs_sorted[range(-(samp_size + 1), 0)]
        diff_probs = np.ediff1d(last_nplus1_probs)
        s = np.sum(probs_sorted[0 : (pop_size - samp_size)])
        initial_probs_selection = (
            samp_size
            * diff_probs
            * (1 + np.linspace(1, samp_size, samp_size) * probs_sorted[pop_size - samp_size] / s)
        )
        probs_sorted = np.delete(probs_sorted, -1)
        selected_i = np.random.choice(np.arange(0, samp_size), size=1, p=initial_probs_selection)[
            0
        ]
        sampled_indices = all_indices_sorted[selected_i + 1 : samp_size]

        notsampled_indices = np.delete(all_indices_sorted, sampled_indices)
        notsampled_probs = np.delete(probs_sorted, sampled_indices)
        p_denominator = (
            s
            + np.linspace(
                1,
                pop_size - samp_size + selected_i + 1,
                pop_size - samp_size + selected_i + 1,
            )
            * probs_sorted[pop_size - samp_size]
        )
        p_starts = notsampled_probs / p_denominator
        range_part2 = range(pop_size - samp_size, pop_size - samp_size + selected_i)
        p_starts[range_part2] = probs_sorted[pop_size - samp_size] / p_denominator[range_part2]
        p_starts_sum = np.cumsum(np.flip(p_starts)[range(p_starts.size - 1)])
        p_starts_sum = np.append(np.flip(p_starts_sum), 1)
        p_double_starts = p_starts / p_starts_sum
        p_double_starts[-1] = 0

        start_j = 0
        end_j = pop_size - samp_size + 1
        for ll in np.arange(selected_i + 1):
            sampling_space = range(start_j, end_j)
            p_double_space = p_double_starts[range(start_j, end_j)]
            p_double_space = 1 - (selected_i + 1 - ll) * np.append(0, p_double_space)
            p_double_space = np.delete(p_double_space, -1)
            a_j = (
                (samp_size - ll + 1) * p_starts[range(start_j, end_j)] * np.cumprod(p_double_space)
            )
            indice_j = np.random.choice(sampling_space, size=1, p=a_j / np.sum(a_j))[0]
            selected_j = notsampled_indices[indice_j]
            sampled_indices = np.append(sampled_indices, selected_j)
            start_j = indice_j + 1
            end_j += 1
        sample = hits = np.zeros(samp_unit.size).astype(int)
        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    @staticmethod
    def _pps_brewer_select(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        all_indices = np.arange(samp_unit.size)
        all_probs = mos / np.sum(mos)
        working_probs = all_probs * (1 - all_probs) / (1 - samp_size * all_probs)
        working_probs = working_probs / np.sum(working_probs)
        sampled_indices = np.random.choice(all_indices, 1, p=working_probs)
        sample = hits = np.zeros(samp_unit.size).astype(int)
        for s in np.arange(1, samp_size):
            remaining_indices = np.delete(all_indices, sampled_indices)
            remaining_probs = np.delete(all_probs, sampled_indices)
            remaining_probs = (
                remaining_probs * (1 - remaining_probs) / (1 - (samp_size - s) * remaining_probs)
            )
            remaining_probs = remaining_probs / sum(remaining_probs)
            current_selection = np.random.choice(remaining_indices, 1, p=remaining_probs)
            sampled_indices = np.append(sampled_indices, current_selection)

        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    @staticmethod
    def _pps_murphy_select(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        if samp_size != 2:
            raise ValueError(
                "For the Murphy's selection algorithm, sample size must be equal to 2"
            )
        all_indices = np.arange(samp_unit.size)
        all_probs = mos / np.sum(mos)
        sampled_indices = np.random.choice(all_indices, 1, p=all_probs)
        remaining_indices = np.delete(all_indices, sampled_indices)
        remaining_probs = np.delete(all_probs, sampled_indices)
        remaining_probs = remaining_probs / (1 - all_probs[sampled_indices])
        current_selection = np.random.choice(remaining_indices, 1, p=remaining_probs)
        sampled_indices = np.append(sampled_indices, current_selection)

        sample = hits = np.zeros(samp_unit.size).astype(int)
        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    @staticmethod
    def _pps_rs_select(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        all_indices = np.arange(samp_unit.shape[0])
        all_probs = mos / np.sum(mos)

        stop = False
        sample = hits = np.zeros(samp_unit.shape[0]).astype(int)
        sampled_indices = None
        while stop is not True:
            sampled_indices = np.random.choice(all_indices, 1, p=all_probs)
            remaining_indices = np.delete(all_indices, sampled_indices)
            remaining_probs = all_probs / (1 - samp_size * all_probs)
            remaining_probs = np.delete(remaining_probs, sampled_indices)
            remaining_probs = remaining_probs / np.sum(remaining_probs)
            remaining_sample = np.random.choice(
                remaining_indices, samp_size - 1, p=remaining_probs
            )
            _, counts = np.unique(np.append(sampled_indices, remaining_sample), return_counts=True)
            if (counts == 1).all():
                stop = True

        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    def _pps_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        samp_unit = formats.sample_units(samp_unit, unique=True)
        samp_size = formats.sample_size_dict(samp_size, self.strat, stratum)

        sample = hits = np.zeros(samp_unit.size).astype(int)
        if isinstance(samp_size, dict):
            for s in np.unique(stratum):
                stratum_units = stratum == s
                if self.method == SelectMethod.pps_sys:  # systematic
                    (sample[stratum_units], hits[stratum_units],) = self._pps_sys_select(
                        samp_unit[stratum_units],
                        samp_size[s],
                        mos[stratum_units],
                    )
                elif self.method == SelectMethod.pps_hv:  # "hanurav-vijayan"
                    (sample[stratum_units], hits[stratum_units],) = self._pps_hv_select(
                        samp_unit[stratum_units],
                        samp_size[s],
                        mos[stratum_units],
                    )
                elif self.method == SelectMethod.pps_brewer:
                    (sample[stratum_units], hits[stratum_units],) = self._pps_brewer_select(
                        samp_unit[stratum_units],
                        samp_size[s],
                        mos[stratum_units],
                    )
                elif self.method == SelectMethod.pps_murphy:
                    (sample[stratum_units], hits[stratum_units],) = self._pps_murphy_select(
                        samp_unit[stratum_units],
                        samp_size[s],
                        mos[stratum_units],
                    )
                elif self.method == SelectMethod.pps_rs:
                    (sample[stratum_units], hits[stratum_units],) = self._pps_rs_select(
                        samp_unit[stratum_units],
                        samp_size[s],
                        mos[stratum_units],
                    )
        else:
            if self.method == SelectMethod.pps_sys:  # systematic
                sample, hits = self._pps_sys_select(samp_unit, samp_size, mos)
            elif self.method == SelectMethod.pps_hv:  # "hanurav-vijayan"
                sample, hits = self._pps_hv_select(samp_unit, samp_size, mos)
            elif self.method == SelectMethod.pps_brewer:
                sample, hits = self._pps_brewer_select(samp_unit, samp_size, mos)
            elif self.method == SelectMethod.pps_murphy:
                sample, hits = self._pps_murphy_select(samp_unit, samp_size, mos)
            elif self.method == SelectMethod.pps_rs:
                sample, hits = self._pps_rs_select(samp_unit, samp_size, mos)

        return sample, hits

    # SYSTEMATIC methods

    def _sys_inclusion_probs(
        self,
        samp_unit: np.ndarray,
        samp_size: Optional[Union[DictStrInt, int]] = None,
        stratum: Optional[np.ndarray] = None,
        samp_rate: Optional[Union[DictStrFloat, float]] = None,
    ) -> np.ndarray:

        pass

    @staticmethod
    def _sys_selection_method(
        samp_unit: np.ndarray,
        samp_size: Optional[int],
        samp_rate: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray]:

        if samp_size is not None and samp_rate is not None:
            raise AssertionError(
                """Both samp_size and samp_rate are provided. 
                Only one of the two parameters should be specified."""
            )

        if samp_rate is None and samp_size is None:
            raise AssertionError("samp_size or samp_rate must be provided!")

        if samp_rate is not None:
            samp_size = int(samp_rate * samp_unit.size)
        samp_interval = int(samp_unit.size / samp_size)  # same as 1 / samp_rate

        random_start = np.random.choice(range(0, samp_interval))
        random_picks = random_start + samp_interval * np.linspace(
            0, samp_size - 1, samp_size
        ).astype(int)
        hits = np.zeros(samp_unit.size).astype(int)
        hits[random_picks] = 1

        return hits == 1, hits

    def _sys_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Optional[Union[DictStrInt, int]] = None,
        stratum: Optional[np.ndarray] = None,
        samp_rate: Union[DictStrFloat, float, None] = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        sample = hits = np.zeros(samp_unit.size).astype(int)
        if self.strat and (isinstance(samp_size, dict) or isinstance(samp_rate, dict)):
            for s in np.unique(stratum):
                samp_size_s = None
                if samp_size is not None:
                    samp_size_s = samp_size[s]
                samp_rate_s = None
                if samp_rate is not None:
                    samp_rate_s = samp_rate[s]
                stratum_units = stratum == s
                (
                    sample[stratum_units],
                    hits[stratum_units],
                ) = self._sys_selection_method(samp_unit[stratum_units], samp_size_s, samp_rate_s)
        elif isinstance(samp_size, int) or isinstance(samp_rate, float):
            samp_size_n = None if samp_size is None else samp_size
            samp_rate_n = None if samp_rate is None else samp_rate
            sample, hits = self._sys_selection_method(samp_unit, samp_size_n, samp_rate_n)

        return sample, hits

    def inclusion_probs(
        self,
        samp_unit: Array,
        samp_size: Optional[Union[DictStrInt, int]] = None,
        stratum: Optional[Array] = None,
        mos: Optional[Array] = None,
        samp_rate: Optional[Union[DictStrFloat, float]] = None,
    ) -> np.ndarray:

        if samp_size is not None and samp_rate is not None:
            raise AssertionError(
                """Both samp_size and samp_rate are provided. 
                Only one of the two parameters should be specified."""
            )

        if self.strat and stratum is None:
            raise AssertionError("Stratum must be provided for stratified samples!")

        samp_unit = formats.sample_units(samp_unit, unique=True)

        samp_size_temp: Union[DictStrInt, int]
        if stratum is not None:
            stratum = formats.numpy_array(stratum)
            if isinstance(samp_size, (int, float)):
                strata = np.unique(stratum)
                samp_size_temp = dict(zip(strata, np.repeat(int(samp_size), strata.shape[0])))
            elif isinstance(samp_size, dict):
                samp_size_temp = samp_size.copy()
            else:
                raise TypeError("samp_size or samp_rate has the wrong type")
        else:
            if isinstance(samp_size, (int, float)):
                samp_size_temp = int(samp_size)
            else:
                raise TypeError("samp_size or samp_rate has the wrong type")

        mos = formats.numpy_array(mos) if mos is not None else np.ones(samp_unit.shape[0])

        samp_size_temp = formats.sample_size_dict(samp_size_temp, self.strat, stratum)

        if self.method == SelectMethod.srs:
            incl_probs = self._srs_inclusion_probs(samp_unit, samp_size, stratum)
        elif self.method in (
            SelectMethod.pps_sys,
            SelectMethod.pps_hv,
            SelectMethod.pps_brewer,
            SelectMethod.pps_murphy,
            SelectMethod.pps_rs,
        ):
            if self._anycertainty(samp_size_temp, stratum, mos):
                raise AssertionError("Some clusters are certainties.")
            incl_probs = self._pps_inclusion_probs(samp_unit, samp_size_temp, mos, stratum)
        elif self.method == SelectMethod.sys:
            incl_probs = self._sys_inclusion_probs(samp_unit, samp_size_temp, stratum, samp_rate)
        else:
            raise ValueError("method not valid!")

        return incl_probs

    def joint_inclusion_probs(self) -> None:
        pass

    def select(
        self,
        samp_unit: Array,
        samp_size: Optional[Union[DictStrInt, int]] = None,
        stratum: Optional[Array] = None,
        mos: Optional[Array] = None,
        samp_rate: Optional[Union[DictStrFloat, float]] = None,
        probs: Optional[Array] = None,
        shuffle: bool = False,
        to_dataframe: bool = False,
        sample_only: bool = False,
        remove_nan: bool = False,
    ) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:

        if samp_size is not None and samp_rate is not None:
            raise AssertionError(
                """Both samp_size and samp_rate are provided. 
                Only one of the two parameters should be specified."""
            )

        if self.strat and stratum is None:
            raise AssertionError("Stratum must be provided for stratified samples!")

        samp_unit = formats.sample_units(samp_unit, unique=True)
        mos = formats.numpy_array(mos) if mos is not None else np.ones(samp_unit.shape[0])
        if probs is not None:
            probs = formats.numpy_array(probs)

        samp_size_temp: Union[DictStrInt, int]
        samp_rate_temp: Union[DictStrFloat, float]

        if stratum is not None:
            stratum = formats.numpy_array(stratum)
            if isinstance(samp_size, (int, float)):
                strata = np.unique(stratum)
                samp_size_temp = dict(zip(strata, np.repeat(samp_size, strata.shape[0])))
            elif isinstance(samp_size, dict):
                samp_size_temp = samp_size.copy()
            elif isinstance(samp_rate, (int, float)):
                strata = np.unique(stratum)
                samp_rate_temp = dict(zip(strata, np.repeat(samp_rate, strata.shape[0])))
            elif isinstance(samp_rate, dict):
                samp_rate_temp = samp_rate.copy()
            else:
                raise TypeError("samp_size or samp_rate has the wrong type")
        else:
            if isinstance(samp_size, (int, float)):
                samp_size_temp = int(samp_size)  # {"__dummy__": samp_size}
                samp_rate_temp = None
            elif isinstance(samp_rate, (int, float)):
                samp_rate_temp = float(samp_rate)
                samp_size_temp = None
            else:
                raise TypeError("samp_size or samp_rate has the wrong type")


        suffled_order = None
        if shuffle and self.method in (SelectMethod.sys, SelectMethod.pps_sys):
            suffled_order = np.linspace(0, samp_unit.size - 1, samp_unit.size).astype(int)
            np.random.shuffle(suffled_order)
            samp_unit = samp_unit[suffled_order]
            if stratum is not None:
                stratum = stratum[suffled_order]
            if self.method == SelectMethod.pps_sys and mos is not None:
                mos = mos[suffled_order]

        if self.method == SelectMethod.srs and samp_size_temp is not None:
            probs = self._srs_inclusion_probs(
                samp_unit=samp_unit, samp_size=samp_size_temp, stratum=stratum
            )
            sample, hits = self._grs_select(
                probs=probs, samp_unit=samp_unit, samp_size=samp_size_temp, stratum=stratum
            )
        elif self.method == SelectMethod.sys and samp_rate_temp is not None:
            # probs = self._sys_inclusion_probs(
            #     samp_unit=samp_unit, samp_rate=samp_rate_temp, stratum=stratum
            # )
            sample, hits = self._sys_select(
                samp_unit=samp_unit, samp_size=None, samp_rate=samp_rate_temp, stratum=stratum
            )
        elif self.method in (
            SelectMethod.pps_sys,
            SelectMethod.pps_hv,
            SelectMethod.pps_brewer,
            SelectMethod.pps_murphy,
            SelectMethod.pps_rs,
        ):
            if self._anycertainty(samp_size_temp, stratum, mos):
                raise AssertionError("Some clusters are certainties.")
            probs = self.inclusion_probs(
                samp_unit=samp_unit, samp_size=samp_size_temp, stratum=stratum, mos=mos
            )
            sample, hits = self._pps_select(
                samp_unit=samp_unit, samp_size=samp_size_temp, stratum=stratum, mos=mos
            )
        elif self.method == SelectMethod.sys and samp_rate_temp is None:
            # probs = self._srs_inclusion_probs(samp_unit, samp_size, stratum) - Todo
            sample, hits = self._sys_select(
                samp_unit=samp_unit,
                samp_size=samp_size_temp,
                stratum=stratum,
                samp_rate=None,
            )
        elif self.method == SelectMethod.grs:
            if probs is None:
                raise ValueError("Probabilities of inclusion not provided!")
            else:
                sample, hits = self._grs_select(
                    probs, samp_unit, samp_size=samp_size_temp, stratum=stratum
                )
        else:
            raise ValueError("sampling method not implemented!")

        if shuffle:
            sample = sample[suffled_order]
            hits = hits[suffled_order]

        if to_dataframe and sample_only:
            frame = self._to_dataframe(
                samp_unit=samp_unit,
                stratum=stratum,
                mos=mos,
                sample=sample,
                hits=hits,
                probs=probs,
            )
            return frame.loc[frame["_sample"] == 1].reset_index(drop=True)
        elif to_dataframe and not sample_only:
            frame = self._to_dataframe(
                samp_unit=samp_unit,
                stratum=stratum,
                mos=mos,
                sample=sample,
                hits=hits,
                probs=probs,
            )
            return frame
        elif not to_dataframe and sample_only and probs is not None:
            return sample[sample], hits[sample], probs[sample]
        else:
            return sample, hits, probs
