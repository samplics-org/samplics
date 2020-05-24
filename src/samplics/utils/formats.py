"""Formats to tranform the data into another format.

Functions:
    | *numpy_array()* converts an array-like data to np.ndarray
    | *array_to_dict()* converts an array to a dictionary where the keys are the unique values of 
    |   the array and the values of the dictionary are the counts of the array values. 
    | *dataframe_to_array()* returns a pandas dataframe from an np.ndarray.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from samplics.utils import checks

from samplics.utils.types import Array, Number, StringNumber, DictStrNum


def numpy_array(arr: Any) -> np.ndarray:
    """Converts an array-like input data to np.ndarray. 

    Args:
        arr (Any): array-like input data.

    Returns:
        np.ndarray: an numpy array
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


def sample_size_dict(
    sample_size: Union[Dict[Any, int], int], stratification: bool, stratum: Array,
) -> Dict[Any, int]:
    if not isinstance(sample_size, Dict) and stratification:
        strata = np.unique(stratum)
        samp_dict = dict(zip(strata, np.repeat(sample_size, strata.size)))
    if not isinstance(sample_size, Dict) and not stratification:
        samp_dict = {"__none__": sample_size}
    elif isinstance(sample_size, Dict):
        samp_dict = sample_size

    return samp_dict


def sample_units(all_units: Array, unique: bool = True) -> np.ndarray:
    all_units = numpy_array(all_units)
    if unique:
        checks.assert_not_unique(all_units)

    return all_units


def dict_to_dataframe(col_names: List[str], *args: Dict[Any, Number]) -> pd.DataFrame:

    keys = list(args[0].keys())
    number_keys = len(keys)
    values = []
    for k, arg in enumerate(args):
        argk_keys = list(args[k].keys())
        if not isinstance(arg, dict) or (keys != argk_keys) or number_keys != len(argk_keys):
            raise AssertionError("All input parameters must be dictionaries with the same keys.")

        values.append(list(arg.values()))

    values_df = pd.DataFrame(values,).T
    values_df.insert(0, "00", keys)
    values_df.columns = col_names

    return values_df
