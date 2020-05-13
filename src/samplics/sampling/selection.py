from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd

import math

from samplics.utils import checks, formats
from samplics.utils.types import Array, Number, StringNumber


class Sample:
    """Class implements an arbitrary random selection based on given 
    inclusion probability and sample size

    IDEA: for given probs and sample size, select a sample. 
    It can be stratified or not.
    """

    def __init__(
        self, method: str, stratification: bool = False, with_replacement: bool = True,
    ) -> None:
        if method.lower() in (
            "srs",
            "sys",
            "pps-brewer",
            "pps-hv",  # Hanurav-Vijayan
            "pps-murphy",
            "pps-sampford",
            "pps-sys",
            "grs",
        ):
            self.method = method.lower()  # grs for generic random selection
        else:
            raise ValueError(
                "method must be: 'srs', 'sys', 'pps-brewer', 'pps-hv', 'pps-murphy', 'pps-sampford, 'pps-sys' or 'grs'"
            )
        self.stratification = True if stratification else False
        self.with_replacement = with_replacement
        self.fpc: Dict[Any, float] = {}

    @staticmethod
    def _convert_to_dict(obj_any: Any, obj_type: type) -> Dict[Any, Any]:

        if obj_any is not None and isinstance(obj_any, obj_type):
            obj_dict = {"__none__": obj_any}
        elif obj_any is not None and isinstance(obj_any, Dict):
            obj_dict = obj_any
        elif not isinstance(obj_any, dict):
            raise TypeError(f"{str(obj_any)} must be a dictionary or {obj_type.__name__}.")

        return obj_dict

    @staticmethod
    def _to_dataframe(
        samp_unit: np.ndarray,
        stratum: np.ndarray,
        mos: np.ndarray,
        sample: np.ndarray,
        hits: np.ndarray,
        probs: np.ndarray,
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

        if stratum is None:
            df.drop(columns=["_stratum"], inplace=True)
        if mos is None:
            df.drop(columns=["_mos"], inplace=True)

        return df

    def _calculate_fpc(
        self, samp_unit: np.ndarray, samp_size: Union[Dict[Any, int], int], stratum: np.ndarray,
    ) -> None:

        samp_unit = checks.check_sample_unit(samp_unit)
        samp_size = checks.check_sample_size_dict(samp_size, self.stratification, stratum)

        self.fpc = dict()
        if self.stratification:
            strata = np.unique(stratum)
            for k, s in enumerate(strata):
                number_units_s = len(samp_unit[stratum == s])
                self.fpc[s] = np.sqrt((number_units_s - samp_size[s]) / (number_units_s - 1))
        else:
            self.fpc["__none__"] = np.sqrt(
                (samp_unit.size - samp_size["__none__"]) / (samp_unit.size - 1)
            )

    def _grs_select(
        self,
        probs: np.ndarray,
        samp_unit: np.ndarray,
        samp_size: Union[Dict[Any, int], int],
        stratum: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        select a sample. 

        Args:
            samp_unit (array) : Array of the units from the sampling 
            frame. This vector should contains unique values.   

            probs (integer) : Array of the inclusion probabilities.   

            samp_size (int or dictionary) : The number of units from
            the frame to select in the sample. For stratified designs, 
            a dictionary providing the pairing between strata and 
            sample sizes will be expected.     

            stratum (array) : An array indicating the stratification 
            for all the units in the sampling frame. This vector has
            the same length as samp_unit.

        Returns:
            A tuple of two arrays: the first array indicates the 
            selected sample and the second array provides the 
            number of hits.
        """

        samp_size = checks.check_sample_size_dict(samp_size, self.stratification, stratum)
        sample = hits = np.zeros(samp_unit.size).astype("int")
        self._calculate_fpc(samp_unit, samp_size, stratum)
        if self.stratification:
            all_indices = np.array(range(samp_unit.size))
            sampled_indices_list = []
            for s in np.unique(stratum):
                stratum_units = stratum == s
                probs_s = probs[stratum_units] / np.sum(probs[stratum_units])
                sampled_indices_s = np.random.choice(
                    all_indices[stratum_units], samp_size[s], self.with_replacement, probs_s,
                )
                sampled_indices_list.append(sampled_indices_s)
                sampled_indices = [val for sublist in sampled_indices_list for val in sublist]
            sampled_indices = np.array(sampled_indices).flatten()
        else:
            sampled_indices = np.random.choice(
                samp_unit.size,
                samp_size["__none__"],
                self.with_replacement,
                probs / np.sum(probs),
            )

        indices_s, hits_s = np.unique(sampled_indices, return_counts=True)
        sample[indices_s] = True
        hits[indices_s] = hits_s

        return sample, hits

    @staticmethod
    def _anycertainty(
        samp_size: Dict[StringNumber, int], stratum: np.ndarray, mos: np.ndarray,
    ) -> bool:

        if stratum is not None:
            probs = np.zeros(stratum.size)
            for s in np.unique(stratum):
                stratum_units = stratum == s
                mos_s = mos[stratum_units]
                probs[stratum_units] = samp_size[s] * mos_s / np.sum(mos_s)
        else:
            probs = samp_size["__none__"] * mos / np.sum(mos)

        certainty: bool = (probs >= 1).any()

        return certainty

    # SRS methods
    def _srs_inclusion_probs(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[Dict[Any, int], int],
        stratum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        The inclusion probabilities based on the simple random 
        selection (SRS) sampling approach

        Args:
            samp_unit (array) : Array of the units from the sampling 
            frame. This vector should contains unique values.      

            samp_size (integer or dictionary) : The size of the 
            sample to be selected from the sampling frame. 
            For stratified design, samp_size is a dictionary 
            providing the sample sizes for the strata.       

            stratum (array) : An array indicating the stratification 
            for all the units in the sampling frame. This vector has 
            the same length as samp_unit.
        
        Returns:
            A array of the inclusion probabilities
        """

        samp_unit = checks.check_sample_unit(samp_unit)
        samp_size = checks.check_sample_size_dict(samp_size, self.stratification, stratum)

        number_units = samp_unit.size
        if self.stratification:
            incl_probs = np.zeros(number_units) * np.nan
            for s in np.unique(stratum):
                number_units_s = samp_unit[stratum == s].size
                incl_probs[stratum == s] = samp_size[s] / number_units_s
        else:
            number_units = samp_unit.size
            incl_probs = np.ones(number_units) * samp_size["__none__"] / number_units

        return incl_probs

    # PPS methods
    def _pps_inclusion_probs(
        self,
        samp_unit: np.ndarray,
        samp_size: Dict[Any, int],
        mos: np.ndarray,
        stratum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        The inclusion probabilities based on the simple random 
        selection (SRS) sampling approach

        Args: 
            samp_unit (array) : Array of the units from the sampling 
            frame. This vector should contains unique values.      

            samp_size (int or dictionary) : The number of units from 
            the frame to select in the sample. For stratified designs, 
            a dictionary providing the pairing between strata and 
            sample sizes will be expected.              

            mos (array) : The size associated to each sampling 
            unit in the frame. 

            stratum (array) : An array indicating the stratification 
            for all the units in the sampling frame. This vector has 
            the same length as samp_unit.

        Returns: 
            A array of the inclusion probabilities
        """

        samp_unit = checks.check_sample_unit(samp_unit)
        samp_size = checks.check_sample_size_dict(samp_size, self.stratification, stratum)

        if self.stratification:
            number_units = samp_unit.size
            incl_probs = np.zeros(number_units) * np.nan
            for s in np.unique(stratum):
                stratum_units = stratum == s
                mos_s = mos[stratum_units]
                incl_probs[stratum_units] = samp_size[s] * mos_s / np.sum(mos_s)
        else:
            incl_probs = samp_size["__none__"] * mos / np.sum(mos)

        return incl_probs

    @staticmethod
    def _pps_sys_select(
        samp_unit: np.ndarray, samp_size: int, mos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        cumsize = np.append(0, np.cumsum(mos))
        samp_interval = cumsize[-1] / samp_size
        random_start = np.random.random_sample() * samp_interval
        random_picks = random_start + samp_interval * np.linspace(0, samp_size - 1, samp_size)

        hits = np.zeros(samp_unit.size).astype("int")
        for k in range(cumsize.size - 1):
            for ll in range(random_picks.size):
                if cumsize[k] < random_picks[ll] <= cumsize[k + 1]:
                    hits[k] += 1

        return hits >= 1, hits

    @staticmethod
    def _pps_hv_select(
        samp_unit: np.ndarray, samp_size: int, mos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

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
                1, pop_size - samp_size + selected_i + 1, pop_size - samp_size + selected_i + 1,
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
        sample = hits = np.zeros(samp_unit.size).astype("int")
        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    @staticmethod
    def _pps_brewer_select(
        samp_unit: np.ndarray, samp_size: int, mos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        all_indices = np.arange(samp_unit.size)
        all_probs = mos / np.sum(mos)
        working_probs = all_probs * (1 - all_probs) / (1 - samp_size * all_probs)
        working_probs = working_probs / np.sum(working_probs)
        sampled_indices = np.random.choice(all_indices, 1, p=working_probs)
        sample = hits = np.zeros(samp_unit.size).astype("int")
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
        samp_unit: np.ndarray, samp_size: int, mos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

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

        sample = hits = np.zeros(samp_unit.size).astype("int")
        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    @staticmethod
    def _pps_sampford_select(samp_unit: Array, samp_size: int, mos: Array) -> Tuple[Array, Array]:

        all_indices = np.arange(samp_unit.size)
        all_probs = mos / np.sum(mos)

        stop = False
        sample = hits = np.zeros(samp_unit.size).astype("int")
        while stop != True:
            sampled_indices = np.random.choice(all_indices, 1, p=all_probs)
            remaining_indices = np.delete(all_indices, sampled_indices)
            remaining_probs = all_probs / (1 - samp_size * all_probs)
            remaining_probs = np.delete(remaining_probs, sampled_indices)
            remaining_probs = remaining_probs / np.sum(remaining_probs)
            remaining_sample = np.random.choice(
                remaining_indices, samp_size - 1, p=remaining_probs
            )
            sampled_indices = np.append(sampled_indices, remaining_sample)
            _, counts = np.unique(sampled_indices, return_counts=True)
            if (counts == 1).all():
                stop = True

        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    def _pps_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Dict[Any, int],
        stratum: np.ndarray,
        mos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        select a sample. 

        Args:
            samp_unit (array) : Array of the units from the sampling 
            frame. This vector should contains unique values.        

            mos (array) : The size associated to each sampling 
            unit in the frame.       

            stratum (array) : An array indicating the stratification 
            for all the units in the sampling frame. This vector has 
            the same length as samp_unit.     

            samp_size (int or dictionary) : The number of units from 
            the frame to select in the sample. For stratified designs, 
            a dictionary providing the pairing between strata and sample 
            sizes will be expected.    

            method (string): A string indicating the selection method. 
            The possible selection methods are: systematic selection 
            (sys or systematic), hanurav-vijayan algorithm (h-v or 
            hanurav-vijayan), Brewer's method (brw or brewer), Murphy's 
            method (mpy or murphy). The methods h-v, brw, and mpy are 
            without replacement algorithms. The default option is None 
            which implements the generic numpy.random.choice algorithm. 
        
        Returns:
            A tuple of two arrays: the first array indicates the 
            selected sample and the second array provides the 
            number  of hits.
        """

        samp_unit = checks.check_sample_unit(samp_unit)
        samp_size = checks.check_sample_size_dict(samp_size, self.stratification, stratum)

        sample = hits = np.zeros(samp_unit.size).astype("int")
        if self.stratification:
            for s in np.unique(stratum):
                stratum_units = stratum == s
                if self.method in "pps-sys":  # systematic
                    (sample[stratum_units], hits[stratum_units],) = self._pps_sys_select(
                        samp_unit[stratum_units], samp_size[s], mos[stratum_units],
                    )
                elif self.method in "pps-hv":  # "hanurav-vijayan"
                    (sample[stratum_units], hits[stratum_units],) = self._pps_hv_select(
                        samp_unit[stratum_units], samp_size[s], mos[stratum_units],
                    )
                elif self.method in "pps-brewer":
                    (sample[stratum_units], hits[stratum_units],) = self._pps_brewer_select(
                        samp_unit[stratum_units], samp_size[s], mos[stratum_units],
                    )
                elif self.method in "pps-murphy":
                    (sample[stratum_units], hits[stratum_units],) = self._pps_murphy_select(
                        samp_unit[stratum_units], samp_size[s], mos[stratum_units],
                    )
                elif self.method in "pps-sampford":
                    (sample[stratum_units], hits[stratum_units],) = self._pps_sampford_select(
                        samp_unit[stratum_units], samp_size[s], mos[stratum_units],
                    )
        else:
            if self.method in "pps-sys":  # systematic
                sample, hits = self._pps_sys_select(samp_unit, samp_size["__none__"], mos)
            elif self.method in "pps-hv":  # "hanurav-vijayan"
                sample, hits = self._pps_hv_select(samp_unit, samp_size["__none__"], mos)
            elif self.method in "pps-brewer":
                sample, hits = self._pps_brewer_select(samp_unit, samp_size["__none__"], mos)
            elif self.method in "pps-murphy":
                sample, hits = self._pps_murphy_select(samp_unit, samp_size["__none__"], mos)
            elif self.method in "pps-sampford":
                sample, hits = self._pps_sampford_select(samp_unit, samp_size["__none__"], mos)

        return sample, hits

    # SYSTEMATIC methods

    def _sys_inclusion_probs(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[Dict[Any, int], int, None] = None,
        stratum: np.ndarray = None,
        samp_rate: Union[Dict[Any, float], float, None] = None,
    ) -> np.ndarray:
        """
        The inclusion probabilities based on the simple random 
        selection (SRS) sampling approach

        Parameters
        ----------
            samp_unit: array
                Array of the units from the sampling frame. This vector should contains unique values.

            samp_size:  int or dictionary
                The number of units from the frame to select in the sample. For stratified designs, a dictionary providing the pairing between strata and sample sizes will be expected. 

            stratum : array
                An array indicating the stratification for all the units in the sampling frame. This vector has the same length as samp_unit.

            samp_rate : dictionary or float
                The rate at which the sample is selected from the frame. For stratified design, samp_rate is a dictionary providing the rates for the strata.

        Returns
        -------
            out: array
                A array of the inclusion probabilities
        """

        pass

    @staticmethod
    def _sys_selection_method(
        samp_unit: np.ndarray, samp_size: int, samp_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:

        if samp_size is not None and samp_rate is not None:
            raise AssertionError(
                "Both samp_size and samp_rate are provided. Only one of the two parameters should be specified."
            )

        if samp_rate is not None:
            samp_size = math.floor(samp_rate * samp_unit.size)
        samp_interval = math.floor(samp_unit.size / samp_size)  # same as 1 / samp_rate
        random_start = np.random.choice(range(0, samp_interval))
        random_picks = random_start + samp_interval * np.linspace(
            0, samp_size - 1, samp_size
        ).astype("int")
        hits = np.zeros(samp_unit.size).astype("int")
        hits[random_picks] = 1

        return hits == 1, hits

    def _sys_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Dict[Any, int],
        stratum: np.ndarray,
        samp_rate: Dict[Any, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The inclusion probabilities based on the simple random 
        selection (SRS) sampling approach

        Args:
            samp_unit (array) : Array of the units from the sampling 
            frame. This vector should contains unique values.

            samp_size (int or dictionary) : The number of units from 
            the frame to select in the sample. For stratified designs, 
            a dictionary providing the pairing between strata and sample 
            sizes will be expected. 

            stratum (array) : An array indicating the stratification 
            for all the units in the sampling frame. This vector has 
            the same length as samp_unit.

            samp_rate (dictionary or float) : The rate at which 
            the sample is selected from the frame. For stratified 
            design, samp_rate is a dictionary providing the rates 
            for the strata.

        Returns: 
            A array of the inclusion probabilities
        """

        sample = hits = np.zeros(samp_unit.size).astype("int")
        if self.stratification:
            for s in np.unique(stratum):
                samp_size_s = None if samp_size is None else samp_size[s]
                samp_rate_s = None if samp_rate is None else samp_rate[s]
                stratum_units = stratum == s
                (sample[stratum_units], hits[stratum_units],) = self._sys_selection_method(
                    samp_unit[stratum_units], samp_size_s, samp_rate_s
                )
        else:
            samp_size_n = None if samp_size is None else samp_size["__none__"]
            samp_rate_n = None if samp_rate is None else samp_rate["__none__"]
            sample, hits = self._sys_selection_method(samp_unit, samp_size_n, samp_rate_n)

        return sample, hits

    # public methods

    def inclusion_probs(
        self,
        samp_unit: Array,
        samp_size: Union[Dict[Any, int], int],
        stratum: Optional[Array] = None,
        mos: Optional[Array] = None,
        samp_rate: Union[Dict[Any, float], float, None] = None,
    ) -> np.ndarray:
        """
        The inclusion probabilities based on the simple random 
        selection (SRS) sampling approach

        Parameters
        ----------
        samp_unit: array
            Array of the units from the sampling frame. This vector should contains unique values.

        samp_size:  int or dictionary
            The number of units from the frame to select in the sample. For stratified designs, a dictionary providing the pairing between strata and sample sizes will be expected. 

        stratum : array
            An array indicating the stratification for all the units in the sampling frame. This vector has the same length as samp_unit.

        samp_rate : dictionary or float
            The rate at which the sample is selected from the frame. For stratified design, samp_rate is a dictionary providing the rates for the strata.

        Returns
        -------
        out: array
            A array of the inclusion probabilities
        """
        samp_unit = checks.check_sample_unit(samp_unit)

        if stratum is not None:
            stratum = formats.numpy_array(stratum)
        if mos is not None:
            mos = formats.numpy_array(mos)

        samp_size = checks.check_sample_size_dict(samp_size, self.stratification, stratum)

        if samp_size is not None:
            samp_size = self._convert_to_dict(samp_size, int)
        if samp_rate is not None:
            samp_rate = self._convert_to_dict(samp_rate, float)

        if self.method == "srs":
            incl_probs = self._srs_inclusion_probs(samp_unit, samp_size, stratum)
        elif self.method in ("pps-brewer", "pps-hv", "pps-murphy", "pps-sampford", "pps-sys",):
            if self._anycertainty(samp_size, stratum, mos):
                raise AssertionError("Some clusters are certainties.")
            incl_probs = self._pps_inclusion_probs(samp_unit, samp_size, mos, stratum)
        elif self.method == "sys":
            incl_probs = self._sys_inclusion_probs(samp_unit, samp_size, stratum, samp_rate)

        return incl_probs

    def joint_inclusion_probs(self) -> None:
        """
        The inclusion probabilities based on the simple random 
        selection (SRS) sampling approach

        :arg samp_unit: Array of the units from the sampling frame. This vector should contains unique values.
        :type samp_unit: array

        :arg samp_size: The number of units from the frame to select in the sample. For stratified designs, a dictionary providing the pairing between strata and sample sizes will be expected. 
        :type samp_size: int or dictionary

        :arg stratum: An array indicating the stratification for all the units in the sampling frame. This vector has the same length as samp_unit.
        :type stratum: array

        :arg samp_rate: The rate at which the sample is selected from the frame. For stratified design, samp_rate is a dictionary providing the rates for the strata.
        :type samp_rate: dictionary or float

        :returns: A array of the inclusion probabilities.
        :rtype: array
                
        """
        pass

    def select(
        self,
        samp_unit: Array,
        samp_size: Union[Dict[Any, int], int, None] = None,
        stratum: Optional[Array] = None,
        mos: Optional[Array] = None,
        samp_rate: Union[Dict[Any, float], float, None] = None,
        probs: Optional[Array] = None,
        shuffle: bool = False,
        to_dataframe: bool = False,
        sample_only: bool = False,
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        select a sample. 

        Args:
            samp_unit (array) : Array of the units from the sampling frame. This vector should contains unique values. 
            probs (integer) : Array of the inclusion probabilities.
            samp_size (int or dictionary) : The number of units from the frame to select in the sample. For stratified designs, a dictionary providing the pairing between strata and 
            sample sizes will be expected.
            stratum (array) : An array indicating the stratification for all the units in the sampling frame. This vector has the same length as samp_unit.
            samp_rate (float or dictionary) : The rate to sample from the frame. For stratified designs, a dictionary providing the pairing between strata and selection rates will be expected.  
        
        Returns:
            A tuple of two arrays: the first array indicates the selected sample and the second array provides the number of hits.
        """

        samp_unit = checks.check_sample_unit(samp_unit)

        if stratum is not None:
            stratum = formats.numpy_array(stratum)
        if mos is not None:
            mos = formats.numpy_array(mos)
        if probs is not None:
            probs = formats.numpy_array(probs)

        if samp_size is not None and samp_rate is not None:
            raise AssertionError(
                "Both samp_size and samp_rate are provided. Only one of the two parameters should be specified."
            )

        if samp_size is not None:
            samp_size = checks.check_sample_size_dict(samp_size, self.stratification, stratum)
            samp_size = self._convert_to_dict(samp_size, int)
        if samp_rate is not None:
            samp_rate = self._convert_to_dict(samp_rate, float)

        if shuffle and self.method in ("sys", "pps-sys"):
            suffled_order = np.random.shuffle(range(samp_unit.size))
            samp_unit = samp_unit[suffled_order]
            if stratum is not None:
                stratum = stratum[suffled_order]
            if self.method == "pps-sys" and mos is not None:
                mos = mos[suffled_order]

        if self.method == "srs":
            probs = self._srs_inclusion_probs(samp_unit, samp_size, stratum=stratum)
            sample, hits = self._grs_select(probs, samp_unit, samp_size, stratum)
        elif self.method in ("pps-brewer", "pps-hv", "pps-murphy", "pps-sampford", "pps-sys",):
            if self._anycertainty(samp_size, stratum, mos):
                raise AssertionError("Some clusters are certainties.")
            probs = self.inclusion_probs(samp_unit, samp_size, stratum, mos)
            sample, hits = self._pps_select(samp_unit, samp_size, stratum, mos)
        elif self.method == "sys":
            # probs = self._srs_inclusion_probs(samp_unit, samp_size, stratum)
            sample, hits = self._sys_select(samp_unit, samp_size, stratum, samp_rate)
        elif self.method == "grs":
            sample, hits = self._grs_select(probs, samp_unit, samp_size, stratum)

        if shuffle:
            sample = sample[suffled_order]
            hits = hits[suffled_order]

        if sample_only:
            frame = self._to_dataframe(samp_unit, stratum, mos, sample, hits, probs)
            return frame.loc[frame["_sample"] == 1]
        elif to_dataframe:
            frame = self._to_dataframe(samp_unit, stratum, mos, sample, hits, probs)
            return frame
        else:
            return sample, hits, probs
