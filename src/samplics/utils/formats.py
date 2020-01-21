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


def array_to_dict(arr: np.ndarray, domain: np.array = None) -> Dict:

    if domain is None:
        keys, counts = np.unique(numpy_array(arr), return_counts=True)
        out_dict = dict(zip(keys, counts))
    else:
        out_dict = {}
        for d in np.unique(domain):
            arr_d = arr[domain == d]
            keys_d, counts_d = np.unique(numpy_array(arr_d), return_counts=True)
            out_dict[d] = dict(zip(keys_d, counts_d))

    return out_dict


def dataframe_to_array(df: pd.DataFrame) -> np.ndarray:

    nb_vars = df.shape[1]
    varlist = df.columns
    x_concat = df[varlist[0]]
    if nb_vars > 1:
        for k in range(1, nb_vars):
            x_concat = x_concat.astype(str) + "_&_" + df[varlist[k]].astype(str)
    x_concat.rename(columns="_array", inplace=True)

    return x_concat
