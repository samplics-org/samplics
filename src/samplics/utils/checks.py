import numpy as np
import pandas as pd

from samplics.utils import formats


def _assert_probabilities(probabilities):
    if probabilities.any() > 1:
        raise ValueError("Probabilities cannot be greater than 1")
    elif probabilities.any() < 0:
        raise ValueError("Probabilities must be positive")


def _assert_weights(weights):
    if weights.any() < 0:
        raise ValueError("Sample weights must be positive values")


def _not_unique(array_unique_values):
    if np.unique(array_unique_values).size != len(array_unique_values):
        raise AssertionError(
            "array_unique_values must contains unique values identifying all the units in the sampling frame."
        )


def check_sample_unit(all_units):
    all_units = formats.numpy_array(all_units)
    _not_unique(all_units)

    return all_units


def check_sample_size_dict(sample_size, stratification, stratum):
    if not isinstance(sample_size, dict) and stratification:
        strata = np.unique(stratum)
        return dict(zip(strata, np.repeat(sample_size, strata.size)))
    else:
        return sample_size


def check_response_status(response_status, response_dict):
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


def _check_brr_number_psus(psu):
    if psu.size % 2 != 0:
        raise AssertionError("For the BRR method, the number of PSUs must be a multiple of two.")
