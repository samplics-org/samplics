from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from samplics.utils.types import Array, Number, StringNumber, DictStrNum


def numpy_array(arr: Any) -> np.ndarray:

    if not isinstance(arr, np.ndarray):
        return np.asarray(arr)
    else:
        return arr


def non_missing_array(arr: np.ndarray) -> np.ndarray:

    if not isinstance(arr, np.ndarray):
        return np.asarray(arr)
    else:
        return arr


def array_to_dict(arr: np.ndarray, domain: np.ndarray = None) -> Dict[StringNumber, Number]:

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

    if isinstance(df, pd.Series):
        x_concat = df
    elif isinstance(df, pd.DataFrame):
        nb_vars = df.shape[1]
        varlist = df.columns
        x_concat = df[varlist[0]]
        if nb_vars > 1:
            for k in range(1, nb_vars):
                x_concat = x_concat.astype(str) + "_&_" + df[varlist[k]].astype(str)
    else:
        raise AssertionError("The input data is not a pandas dataframe")

    x_concat.rename(columns="_array", inplace=True)

    return x_concat
