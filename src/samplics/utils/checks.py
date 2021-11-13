"""Runs checks on the data to help capture some errors faster and provides better error messages. 

Functions:
    | *assert_probabilities()* ensures that probability values are between 0 and 1. 
    | *assert_weights()* ensures that sample weights are not negatives. 
    | *assert_not_unique()* return an assertion error if the array has non unique values. 
    | *assert_response_status()* checks that the response values are in ("in", "rr", "nr", "uk").
    | *assert_brr_number_psus()* checks that the number of psus is a multiple of 2. 
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from samplics.utils import formats
from samplics.utils.types import Array, StringNumber


def assert_probabilities(probs: Array) -> None:

    err_msg = "Probabilities must be between 0 and 1, inclusively!"

    if isinstance(probs, (int, float)):
        if probs > 1 or probs < 0:
            raise ValueError(err_msg)
    elif isinstance(probs, (np.ndarray, pd.Series)):
        if probs.any() > 1 or probs.any() < 0:
            raise ValueError(err_msg)


def assert_proportions(probs: Array) -> None:

    err_msg = "Probabilities must be between 0 and 1, inclusively!"

    if isinstance(probs, (int, float)):
        if probs > 1 or probs < 0:
            raise ValueError(err_msg)
    elif isinstance(probs, (np.ndarray, pd.Series)):
        if probs.any() > 1 or probs.any() < 0:
            raise ValueError(err_msg)


def assert_weights(weights: Array) -> None:
    weights = formats.numpy_array(weights)
    if weights.any() < 0:
        raise ValueError("Sample weights must be positive values")


def assert_not_unique(array_unique_values: Array) -> None:
    if np.unique(array_unique_values).size != len(array_unique_values):
        raise AssertionError(
            "array_unique_values must contains unique values identifying all the units in the sampling frame."
        )


def assert_response_status(
    response_status: Any, response_dict: Optional[dict[str, StringNumber]]
) -> None:
    if response_status is None:
        raise AssertionError("response_status is not provided")
    elif not np.isin(response_status, ("in", "rr", "nr", "uk")).all() and response_dict is None:
        raise AssertionError(
            "The response status must only contains values in ('in', 'rr', 'nr', 'uk') or the mapping should be provided using response_dict parameter"
        )
    elif isinstance(response_dict, dict):
        resp_keys = np.array(list(response_dict.keys()))
        if not np.isin(resp_keys, ("in", "rr", "nr", "uk")).all():
            raise AssertionError("Response mapping dictionnary has unexpected value(s)")


def assert_brr_number_psus(psu: np.ndarray) -> None:
    if psu.size % 2 != 0:
        raise AssertionError("For the BRR method, the number of PSUs must be a multiple of two.")
