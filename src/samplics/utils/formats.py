"""Formats to tranform the data into another format.

Functions:
    | *numpy_array()* converts an array-like data to np.ndarray
    | *array_to_dict()* converts an array to a dictionary where the keys are the unique values of 
    |   the array and the values of the dictionary are the counts of the array values. 
    | *dataframe_to_array()* returns a pandas dataframe from an np.ndarray.
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from samplics.utils.types import Array, Number, StringNumber, DictStrNum


def numpy_array(arr: Any) -> np.ndarray:
    """Converts an array-like input data to np.ndarray. 

    Args:
        arr (Any): array-like input data.

    Returns:
        np.ndarray: an numoy array
    """

    if not isinstance(arr, np.ndarray):
        return np.asarray(arr)
    else:
        return arr


def array_to_dict(arr: np.ndarray, domain: np.ndarray = None) -> Dict[StringNumber, Number]:
    """Converts an array to a dictionary where the keys are the unique values of the array and 
    the values of the dictionary are the counts of the array values. 

    Args:
        arr (np.ndarray): an input area.
        domain (np.ndarray, optional): an array to provide the group associated with the 
            observations in arr. If not None, a dictionarry of dictionaries is produced. 
            Defaults to None.

    Returns:
        Dict[StringNumber, Number]: a dictionary with the unique values of *arr* as keys. 
            The values of the dictionary correspond to the counts of the keys in *arr*.
    """

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
    """Returns a pandas dataframe from an np.ndarray. 

    Args:
        df (pd.DataFrame): a pandas dataframe or series.

    Raises:
        AssertionError: return an exception if data is not a pandas dataframe or series.

    Returns:
        np.ndarray: an numpy array.
    """

    if isinstance(df, pd.Series):
        x_array = df
    elif isinstance(df, pd.DataFrame):
        nb_vars = df.shape[1]
        varlist = df.columns
        x_array = df[varlist[0]]
        if nb_vars > 1:
            for k in range(1, nb_vars):
                x_array = x_array.astype(str) + "_&_" + df[varlist[k]].astype(str)
    else:
        raise AssertionError("The input data is not a pandas dataframe")

    x_array.rename(columns="_array", inplace=True)

    return x_array
