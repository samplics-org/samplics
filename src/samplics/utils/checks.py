"""Runs checks on the data to help capture some errors faster and provides better error messages. 

Functions:
    | *assert_probabilities()* ensures that probability values are between 0 and 1. 
    | *assert_weights()* ensures that sample weights are not negatives. 
    | *assert_not_unique()* return an assertion error if the array has non unique values. 
    | *assert_response_status()* checks that the response values are in ("in", "rr", "nr", "uk").
    | *assert_brr_number_psus()* checks that the number of psus is a multiple of 2. 
"""

from typing import TypeVar, Type, Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from samplics.utils import formats
from samplics.utils.types import Array, Number, StringNumber


def assert_probabilities(probabilities: Array) -> None:
    if probabilities.any() > 1:
        raise ValueError("Probabilities cannot be greater than 1")
    elif probabilities.any() < 0:
        raise ValueError("Probabilities must be positive")


def assert_weights(weights: Array) -> None:
    if weights.any() < 0:
        raise ValueError("Sample weights must be positive values")


def assert_not_unique(array_unique_values: Array) -> None:
    if np.unique(array_unique_values).size != len(array_unique_values):
        raise AssertionError(
            "array_unique_values must contains unique values identifying all the units in the sampling frame."
        )


def assert_response_status(response_status: Any, response_dict: Dict[Any, Any]) -> None:
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


def assert_brr_number_psus(psu: np.ndarray) -> None:
    if psu.size % 2 != 0:
        raise AssertionError("For the BRR method, the number of PSUs must be a multiple of two.")
