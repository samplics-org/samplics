from collections import namedtuple
from typing import Protocol


# NumpyArray = TypeVar("np.ndarray", bound=np.ndarray)
# PandasDF = TypeVar("pd.DataFrame", bound=pd.DataFrame)
# PolarsDF = TypeVar("pl.DataFrame", bound=pl.DataFrame)


class DepVars(Protocol):
    y: dict


# Design matrix X (fixed effect) in GLMM
class IndepVarsX(Protocol):
    x: dict


# Design matrix Z (random effect) in GLMM
class IndepVarsZ(Protocol):
    z: dict


class ToDataFrame(Protocol):
    def to_numpy():
        pass

    def to_polars():
        pass

    def to_pandas():
        pass


class GlmmFitStats(Protocol):
    err_stderr: float
    fe_est: namedtuple  # fixed effects
    fe_stderr: namedtuple
    re_stderr: float
    re_stderr_var: float
