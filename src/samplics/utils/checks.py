"""Runs checks on the data to help capture some errors faster and provides better error messages.

Functions:
    | *assert_probabilities()* ensures that probability values are between 0 and 1.
    | *assert_weights()* ensures that sample weights are not negatives.
    | *assert_not_unique()* return an assertion error if the array has non unique values.
    | *assert_response_status()* checks that the response values are in ("in", "rr", "nr", "uk").
    | *assert_brr_number_psus()* checks that the number of psus is a multiple of 2.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from numpy.typing import ArrayLike, NDArray

from samplics.utils import formats
from samplics.utils.types import Array, Number, StringNumber


def assert_probabilities(**kargs: Union[Number, Iterable]) -> None:
    err_msg = "Probabilities must be between 0 and 1, inclusively!"
    for k in kargs:
        if not assert_in_range(0, 1, x=kargs[k]):
            raise ValueError(err_msg)


def assert_proportions(**kargs: Union[Number, Iterable]) -> None:
    err_msg = "Proportions must be between 0 and 1, inclusively!"
    for k in kargs:
        if not assert_in_range(0, 1, x=kargs[k]):
            raise ValueError(err_msg)


def assert_in_range(low: Number, high: Number, x: Union[Number, Iterable]) -> bool:
    if isinstance(x, (int, float)):
        if x > high or x < low:
            return False
    elif isinstance(x, (np.ndarray, pd.Series)):
        if (x > high).any() or (x < low).any():
            return False
    elif isinstance(x, Iterable):
        for i in x:
            if isinstance(x, dict):
                if x[i] > high or x[i] < low:
                    return False
            elif i > high or i < low:
                return False

    return True


def assert_weights(weights: Array) -> None:
    weights = formats.numpy_array(weights)
    if (weights < 0).any():
        raise ValueError("Sample weights must be positive values")


def assert_not_unique(array_unique_values: Array) -> None:
    if np.unique(array_unique_values).size != len(array_unique_values):
        raise AssertionError(
            "The array must contains unique values identifying all the units in the sampling frame."
        )


def assert_response_status(
    response_status: Union[str, np.ndarray],
    response_dict: Optional[dict[str, StringNumber]],
) -> None:
    if response_status is None:
        raise AssertionError("response_status is not provided")
    elif not np.isin(response_status, ("in", "rr", "nr", "uk")).all() and response_dict is None:
        raise AssertionError(
            "The response status must only contains values in ('in', 'rr', 'nr', 'uk') or the mapping should be provided using response_dict parameter"
        )
    elif "rr" not in response_status and response_dict is None:
        raise AssertionError("The response status must at least contains rr!")
    elif isinstance(response_dict, dict):
        # resp_keys = list(response_dict.keys())
        resp_keys = [x.lower() for x in response_dict]
        if not np.isin(resp_keys, ("in", "rr", "nr", "uk")).all():
            raise AssertionError("Response mapping dictionnary has unexpected value(s)")


def assert_brr_number_psus(psu: np.ndarray) -> None:
    if psu.size % 2 != 0:
        raise AssertionError("For the BRR method, the number of PSUs must be a multiple of two.")


def _raise_singleton_error(single_psu_strata: Array) -> None:
    raise ValueError(f"Only one PSU in the following strata: {single_psu_strata}")


def _skip_singleton(single_psu_strata: Array, skipped_strata: Array) -> Array:
    skipped_str = np.isin(single_psu_strata, skipped_strata)
    if skipped_str.sum() > 0:
        return single_psu_strata[skipped_str]
    else:
        raise ValueError("{skipped_strata} does not contain singleton PSUs")


def _certainty_singleton(
    singletons: Array,
    _stratum: Array,
    _psu: Array,
    _ssu: Array,
) -> Array:
    # Make a writable copy (avoids 'assignment destination is read-only')
    psu = np.array(_psu, copy=True)

    # Normalize inputs
    singletons = np.atleast_1d(singletons)
    stratum = np.asarray(_stratum)

    # Case 1: per-record SSU array provided → direct copy for singleton strata
    if np.ndim(_ssu) == 1 and _ssu.shape[0] == stratum.shape[0]:
        ssu = np.asarray(_ssu)
        mask = np.isin(stratum, singletons)
        psu[mask] = ssu[mask]
        return psu

    # Case 2: scalar/0-d SSU → assign sequential PSU IDs within each singleton stratum
    if np.ndim(_ssu) == 0 or (_ssu.shape == () or _ssu.shape == (0,)):
        for s in singletons:
            m = stratum == s
            n = int(np.count_nonzero(m))
            if n > 0:
                # 1,2,...,n as integers (match psu dtype if it's integer)
                seq = np.arange(1, n + 1, dtype=int)
                # Cast to psu dtype if needed
                if psu.dtype.kind in ("i", "u"):
                    seq = seq.astype(psu.dtype, copy=False)
                psu[m] = seq
        return psu

    # Fallback: unexpected _ssu shape
    raise ValueError(
        f"Unexpected _ssu shape {np.shape(_ssu)}; expected scalar or per-record array."
    )


def _combine_strata(
    comb_strata: Mapping[StringNumber, StringNumber],
    _stratum: ArrayLike,
) -> NDArray[np.generic]:
    if not comb_strata:
        raise ValueError("The parameter 'comb_strata' must be a non-empty mapping.")

    # Make a writable copy and preserve dtype
    src = np.asarray(_stratum)
    out = np.array(src, copy=True)

    # Apply mapping; cast new value to out's dtype to avoid dtype surprises
    for old, new in comb_strata.items():
        mask = src == old
        if np.any(mask):
            out[mask] = np.asarray(new, dtype=out.dtype)

    return out
