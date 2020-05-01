# (generated with --quick)

import numpy
from typing import Any, Dict, Type, Union

DictStrNum = Dict[Union[float, int, str], Union[float, int]]

Array: Any
Number: Type[Union[float, int]]
StringNumber: Type[Union[float, int, str]]
np: module
pd: module

def array_to_dict(arr: numpy.ndarray, domain: numpy.ndarray = ...) -> Dict[Union[float, int, str], Union[float, int]]: ...
def dataframe_to_array(df) -> numpy.ndarray: ...
def non_missing_array(arr: numpy.ndarray) -> numpy.ndarray: ...
def numpy_array(arr) -> numpy.ndarray: ...
