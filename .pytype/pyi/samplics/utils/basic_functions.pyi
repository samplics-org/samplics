# (generated with --quick)

import numpy
from typing import Any, Dict, Type, Union

DictStrNum = Dict[Union[float, int, str], Union[float, int]]

Array: Any
Number: Type[Union[float, int]]
StringNumber: Type[Union[float, int, str]]
boxcox: Any
checks: module
formats: module
normal: Any
np: module
pd: module
plt: Any

class BoxCox:
    def _plot_measure(self, y, coef_min = ..., coef_max = ..., nb_points = ..., measure = ...) -> None: ...
    def get_kurtosis(self, y) -> float: ...
    def get_skewness(self, y) -> float: ...
    def plot_kurtosis(self, y, coef_min = ..., coef_max = ..., nb_points = ...) -> None: ...
    def plot_skewness(self, y, coef_min = ..., coef_max = ..., nb_points = ...) -> None: ...
    def transform(self, y, coef: float) -> numpy.ndarray: ...

def sumby(group, y) -> Any: ...
