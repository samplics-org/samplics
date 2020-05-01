# (generated with --quick)

import numpy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

DictStrNum = Dict[Union[float, int, str], Union[float, int]]

Array: Any
Number: Type[Union[float, int]]
StringNumber: Type[Union[float, int, str]]
basic_functions: module
boxcox: Any
checks: module
formats: module
math: module
normal: Any
np: module
pd: module
sm: module

class EbUnitLevel:
    X_s: Any
    Xbar_p: numpy.ndarray
    __doc__: str
    a_factor: Union[dict, numpy.ndarray]
    area_est: dict
    area_mse: Dict[nothing, nothing]
    area_s: numpy.ndarray
    areas_p: Any
    areas_s: Any
    boxcox: Dict[str, Optional[float]]
    constant: Union[float, int]
    convergence: Dict[str, Any]
    error_std: Any
    fe_std: Any
    fitted: bool
    fixed_effects: Any
    gamma: Union[dict, numpy.ndarray]
    goodness: Dict[str, Any]
    indicator: Any
    method: str
    number_reps: int
    number_samples: Optional[int]
    pop_size: numpy.ndarray
    random_effects: numpy.ndarray
    re_std: Union[float, int]
    re_std_cov: Any
    samp_size: Union[dict, numpy.ndarray]
    scale_s: Any
    xbar_s: numpy.ndarray
    y_s: numpy.ndarray
    ybar_s: numpy.ndarray
    def __init__(self, method: str = ..., boxcox: Optional[float] = ..., constant: float = ..., indicator = ...) -> None: ...
    def _predict_indicator(self, number_samples: int, y_s: numpy.ndarray, X_s: numpy.ndarray, area_s: numpy.ndarray, X_r: numpy.ndarray, area_r: numpy.ndarray, areas_r: numpy.ndarray, fixed_effects: numpy.ndarray, gamma: numpy.ndarray, sigma2e: float, sigma2u: float, scale: numpy.ndarray, max_array_length: int, indicator: Callable[..., numpy.ndarray], *args) -> numpy.ndarray: ...
    def _transformation(self, y: numpy.ndarray) -> numpy.ndarray: ...
    def bootstrap_mse(self, number_reps: int, indicator: Callable[..., numpy.ndarray], X: numpy.ndarray, area: numpy.ndarray, scale = ..., intercept: bool = ..., tol: float = ..., maxiter: int = ..., max_array_length: int = ..., *args) -> numpy.ndarray: ...
    def fit(self, y, X, area, samp_weight = ..., scale = ..., intercept: bool = ..., tol: float = ..., maxiter: int = ...) -> None: ...
    def predict(self, number_samples: int, indicator: Callable[..., numpy.ndarray], X: numpy.ndarray, area: numpy.ndarray, samp_weight: Optional[numpy.ndarray] = ..., scale: numpy.ndarray = ..., intercept: bool = ..., max_array_length: int = ..., *args) -> None: ...

class EblupUnitLevel:
    X_s: Any
    Xbar_p: Any
    __doc__: str
    a_factor: dict
    area_est: dict
    area_mse: dict
    area_s: numpy.ndarray
    areas_p: numpy.ndarray
    areas_s: Any
    boxcox: Dict[str, None]
    convergence: Dict[str, Any]
    error_std: Any
    fe_std: Any
    fitted: bool
    fixed_effects: Any
    gamma: dict
    goodness: Dict[str, Any]
    method: str
    number_reps: int
    pop_size: Dict[nothing, nothing]
    random_effects: Any
    re_std: Union[float, int]
    re_std_cov: Any
    samp_size: dict
    scale_s: Any
    xbar_s: numpy.ndarray
    y_s: numpy.ndarray
    ybar_s: numpy.ndarray
    def __init__(self, method: str = ...) -> None: ...
    def _beta(self, y: numpy.ndarray, X: numpy.ndarray, area: numpy.ndarray, weight: numpy.ndarray) -> numpy.ndarray: ...
    def _mse(self, areas: numpy.ndarray, Xs_mean: numpy.ndarray, Xp_mean: numpy.ndarray, gamma: numpy.ndarray, samp_size: numpy.ndarray, scale: numpy.ndarray, A_inv: numpy.ndarray) -> numpy.ndarray: ...
    def _split_data(self, area: numpy.ndarray, X: numpy.ndarray, Xmean: numpy.ndarray, samp_weight: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: ...
    def bootstrap_mse(self, number_reps: int, X: numpy.ndarray, Xmean, area: numpy.ndarray, samp_weight = ..., scale = ..., intercept: bool = ..., tol: float = ..., maxiter: int = ...) -> numpy.ndarray: ...
    def fit(self, y, X, area, samp_weight = ..., scale = ..., intercept: bool = ..., tol: float = ..., maxiter: int = ...) -> None: ...
    def predict(self, Xmean, area, pop_size = ..., intercept: bool = ...) -> None: ...

class EllUnitLevel:
    __doc__: str

class RobustUnitLevel:
    __doc__: str

def area_stats(y: numpy.ndarray, X: numpy.ndarray, area: numpy.ndarray, error_std: float, re_std: float, a_factor: Dict[Any, float], samp_weight: Optional[numpy.ndarray]) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: ...
