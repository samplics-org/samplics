from typing import TypeVar, Type, Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from samplics.utils import formats
from samplics.utils.types import Array, Number, StringNumber


def _assert_probabilities(probabilities: Array) -> None:
    if probabilities.any() > 1:
        raise ValueError("Probabilities cannot be greater than 1")
    elif probabilities.any() < 0:
        raise ValueError("Probabilities must be positive")


def _assert_weights(weights: Array) -> None:
    if weights.any() < 0:
        raise ValueError("Sample weights must be positive values")


def _not_unique(array_unique_values: Array) -> None:
    if np.unique(array_unique_values).size != len(array_unique_values):
        raise AssertionError(
            "array_unique_values must contains unique values identifying all the units in the sampling frame."
        )


def check_sample_unit(all_units: Array) -> np.ndarray:
    all_units = formats.numpy_array(all_units)
    _not_unique(all_units)

    return all_units


def check_sample_size_dict(
    sample_size: Union[Dict[Any, int], int], stratification: bool, stratum: Array,
) -> Dict[Any, int]:
    if not isinstance(sample_size, Dict) and stratification:
        strata = np.unique(stratum)
        samp_size = dict(zip(strata, np.repeat(sample_size, strata.size)))
    if not isinstance(sample_size, Dict) and not stratification:
        samp_size = {"__none__": sample_size}
    elif isinstance(sample_size, Dict):
        samp_size = sample_size

    return samp_size


def check_response_status(response_status: Any, response_dict: Dict[Any, Any]) -> None:
    if response_status is None:
        raise AssertionError("response_status is not provided")
    elif not np.isin(response_status, ("in", "rr", "nr", "uk")).all() and response_dict in (
        None,
        dict(),
    ):
        raise AssertionError(
            "The response status must only contains values in ('in', 'rr', 'nr', 'uk') or the mapping should be provided using response_dict parameter"
        )
    elif response_dict not in (None, dict()):
        resp_keys = np.array(list(response_dict.keys()))
        if not np.isin(resp_keys, ("in", "rr", "nr", "uk")).all():
            raise AssertionError("Response mapping dictionnary has unexpected value(s)")


def _check_brr_number_psus(psu: np.ndarray) -> None:
    if psu.size % 2 != 0:
        raise AssertionError("For the BRR method, the number of PSUs must be a multiple of two.")
