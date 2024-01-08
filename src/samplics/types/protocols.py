from typing import Protocol, runtime_checkable


# NumpyArray = TypeVar("np.ndarray", bound=np.ndarray)
# PandasDF = TypeVar("pd.DataFrame", bound=pd.DataFrame)
# PolarsDF = TypeVar("pl.DataFrame", bound=pl.DataFrame)


@runtime_checkable
class Missing(Protocol):
    ...


@runtime_checkable
class DepVars(Protocol):
    # TODO: Add missing values functionality
    y: dict

    @property
    def nrecords():
        ...

    @property
    def nvars():
        ...

    def to_numpy():
        ...

    def to_polars():
        ...

    def to_pandas():
        ...


@runtime_checkable
class IndepVars(Protocol):
    # TODO: Add missing values functionality
    x: dict

    @property
    def nrecords():
        ...

    @property
    def nvars():
        ...

    def to_numpy():
        ...

    def to_polars():
        ...

    def to_pandas():
        ...


@runtime_checkable
class ToDataFrame(Protocol):
    def to_numpy():
        ...

    def to_polars():
        ...

    def to_pandas():
        ...


# class FitStats(Protocol):
#     method: FitMethod
#     err_stderr: float | dict
#     fe_est: namedtuple  # fixed effects
#     fe_stderr: namedtuple
#     re_stderr: float
#     re_stderr_var: float
