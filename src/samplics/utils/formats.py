from typing import Any, Union, Dict

import numpy as np
import pandas as pd


def numpy_array(arr: Any) -> np.ndarray:

    if not isinstance(arr, np.ndarray):
        return np.asarray(arr)
    else:
        return arr


def _merge_samples(folder, datasets, keys):
    """
    folder provide the location of the datasets
    datasest is a tupple or list of datasets
    keys is the list of merging keys
    """
    pass


def non_missing_array(arr: np.ndarray) -> np.ndarray:

    if not isinstance(arr, np.ndarray):
        return np.asarray(arr)
    else:
        return arr


def array_to_dict(arr: np.ndarray) -> Dict:

    keys, counts = np.unique(numpy_array(arr), return_counts=True)

    return dict(zip(keys, counts))
