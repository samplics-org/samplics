"""Formats to tranform the data into another format.

Functions:
    | *numpy_array()* converts an array-like data to np.ndarray
    | *array_to_dict()* converts an array to a dictionary where the keys are the unique values of 
    |   the array and the values of the dictionary are the counts of the array values. 
    | *dataframe_to_array()* returns a pandas dataframe from an np.ndarray.
"""

from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd

from samplics.utils import checks

from samplics.utils.types import Array, DictStrNum, Number, Series, StringNumber


def numpy_array(arr: Array) -> np.ndarray:
    """Converts an array-like input data to np.ndarray.

    Args:
        arr (Array): array-like input data.

    Returns:
        np.ndarray: an numpy array
    """

    if not isinstance(arr, np.ndarray):
        arr_np = np.asarray(arr)
        if isinstance(arr, (list, tuple)) and len(arr_np.shape) == 2:
            arr_np = np.transpose(arr_np)
        return arr_np
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

    return x_array.to_numpy()


def sample_size_dict(
    sample_size: Union[Dict[StringNumber, Number], Number],
    stratification: bool,
    stratum: Array,
) -> Union[Dict[StringNumber, Number], Number]:
    if not isinstance(sample_size, Dict) and stratification:
        return dict(zip(stratum, np.repeat(sample_size, len(stratum))))
    if isinstance(sample_size, (int, float)) and not stratification:
        return sample_size
    elif isinstance(sample_size, Dict):
        return sample_size


def sample_units(all_units: Array, unique: bool = True) -> np.ndarray:
    all_units = numpy_array(all_units)
    if unique:
        checks.assert_not_unique(all_units)

    return all_units


def dict_to_dataframe(col_names: List[str], *args: Union[DictStrNum, Number]) -> pd.DataFrame:

    if isinstance(args[0], dict):
        keys = list(args[0].keys())
        number_keys = len(keys)
        values = list()
        for k, arg in enumerate(args):
            argk_keys = list(args[k].keys())
            if not isinstance(arg, dict) or (keys != argk_keys) or number_keys != len(argk_keys):
                raise AssertionError(
                    "All input parameters must be dictionaries with the same keys."
                )

            values.append(list(arg.values()))

        values_df = pd.DataFrame(
            values,
        ).T
        values_df.insert(0, "00", keys)
    else:
        values_df = pd.DataFrame({args})

    values_df.columns = col_names

    return values_df


def remove_nans(excluded_units: Array, *args) -> Tuple:

    vars = list()
    for var in args:
        if var is not None and len(var.shape) != 0:
            vars.append(var[~excluded_units])
        else:
            vars.append(None)

    return vars


def fpc_as_dict(stratum: Array, fpc: Union[Array, Number]):

    if stratum is None and isinstance(fpc, (int, float)):
        return fpc
    elif stratum is not None and isinstance(fpc, (int, float)):
        return dict(zip(stratum, np.repeat(fpc, stratum.shape[0])))
    elif stratum is not None and isinstance(fpc, np.ndarray):
        return dict(zip(stratum, fpc))
    else:
        raise TypeError("stratum and fpc are not compatible!")


def concatenate_series_to_str(row: Series) -> str:
    """concatenate the elements into one string using '_' to separate the elements

    Args:
        row (Array): [description]

    Returns:
        str: [description]
    """
    return "__by__".join([str(c) for c in row])


def numpy_to_dummies(arr: np.ndarray, varsnames: List[str]) -> np.ndarray:

    df = pd.DataFrame(arr.astype(str))
    df.columns = varsnames

    return pd.get_dummies(df, drop_first=True)