import datetime as dt
import random as rand

from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl

from attr import validators
from attrs import field, frozen

from samplics.utils.formats import numpy_array
from samplics.utils.types import Array, DictStrNum, Number


# NumpyArray = TypeVar("np.ndarray", bound=np.ndarray)
# PandasDF = TypeVar("pd.DataFrame", bound=pd.DataFrame)
# PolarsDF = TypeVar("pl.DataFrame", bound=pl.DataFrame)


class ToDataFrame(Protocol):
    def to_numpy():
        pass

    def to_polars():
        pass

    def to_pandas():
        pass


@unique
class FitMethod(Enum):
    fh = "FH"
    ml = "ML"
    reml = "REML"


def _is_all_items_positive(obj: Array | DictStrNum) -> bool:
    assert isinstance(obj, (np.ndarray, pd.Series, list, tuple, dict))
    if isinstance(obj, dict):
        arr = numpy_array(list(obj.values()))
    else:
        arr = numpy_array(obj).flatten()

    if (arr <= 0).any():
        return False
    else:
        return True

    return (arr > 0).all()


@frozen
class DirectEst:
    areas: list = field(validator=validators.instance_of(list))
    est: dict = field(validator=validators.instance_of(dict))
    stderr: dict
    ssize: dict
    psize: dict = None
    uid: int = int(
        dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        + str(int(1e16 * rand.random()))
    )

    def __init__(
        self,
        area: Array,
        est: dict | Array | Number,
        stderr: dict | Array | Number,
        ssize: dict | Array | Number,
        psize: dict | Array | Number | None = None,
    ) -> None:
        assert isinstance(area, Array)

        area = numpy_array(area)

        if isinstance(stderr, Number):
            stderr = dict(zip(area, tuple(np.repeat(stderr, len(area)))))
        if isinstance(stderr, Array):
            stderr = dict(zip(area, stderr))

        assert _is_all_items_positive(stderr)

        if isinstance(est, Number):
            est = dict(zip(area, tuple(np.repeat(est, len(area)))))
        if isinstance(est, Array):
            est = dict(zip(area, est))

        if isinstance(ssize, Number):
            ssize = dict(zip(area, tuple(np.repeat(ssize, len(area)))))
        if isinstance(ssize, Array):
            ssize = dict(zip(area, ssize))

        if psize is not None:
            if isinstance(ssize, Number):
                psize = dict(zip(area, tuple(np.repeat(ssize, len(area)))))
            if isinstance(ssize, Array):
                psize = dict(zip(area, stderr))

        self.__attrs_init__(area.tolist(), est, stderr, ssize, psize)

    @property
    def cv(self):
        return {
            key: self.stderr[key] / self.est[key]
            if self.est[key] != 0
            else float("inf") * self.stderr[key]
            for key in self.stderr
        }

    def to_numpy(
        self,
        keep_vars: str | Iterable[str] | None = None,
        drop_vars: str | Iterable[str] | None = None,
    ):
        return self.to_polars(keep_vars=keep_vars, drop_vars=drop_vars).to_numpy()

    def to_polars(
        self,
        keep_vars: str | Iterable[str] | None = None,
        drop_vars: str | Iterable[str] | None = None,
    ):
        aux_df = pl.from_dict(
            {
                "areas": list(self.areas),
                "est": list(self.est.values()),
                "stderr": list(self.stderr.values()),
                "ssize": list(self.ssize.values()),
            }
        )

        if keep_vars is None:
            return aux_df
        elif isinstance(keep_vars, (str, list)):
            return aux_df.select(keep_vars)
        else:
            raise TypeError("varlist must be None or str or list[str]")

    def to_pandas(
        self,
        keep_vars: str | Iterable[str] | None = None,
        drop_vars: str | Iterable[str] | None = None,
    ):
        return self.to_polars(keep_vars=keep_vars, drop_vars=drop_vars).to_pandas()


@frozen
class EblupFit:  # MAYBE call this ModelStats or FitStats or ...
    method: FitMethod
    err_stderr: float
    fe_est: namedtuple  # fixed effects
    fe_stderr: namedtuple
    re_stderr: float
    re_stderr_var: float
    log_llike: float
    convergence: dict
    goodness: dict
    uid: int = int(dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")) + int(
        1e16 * rand.random()
    )

    def to_numpy():  # TODO: To decide if these methods are necessary
        pass

    def to_polars():  # TODO: To decide if these methods are necessary
        pass

    def to_pandas():  # TODO: To decide if these methods are necessary
        pass


@frozen
class EbFit:
    method: FitMethod
    err_stderr: float
    fe_est: namedtuple  # fixed effects
    fe_stderr: namedtuple
    re_stderr: float
    re_stderr_var: float
    log_llike: float
    convergence: dict
    goodness: dict
    uid: int = int(dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")) + int(
        1e16 * rand.random()
    )

    def to_numpy():  # TODO: To decide if these methods are necessary
        pass

    def to_polars():  # TODO: To decide if these methods are necessary
        pass

    def to_pandas():  # TODO: To decide if these methods are necessary
        pass


@frozen
class EblupEst:
    area: list
    est: dict
    fit_stats: EblupFit
    mse: dict | None = None
    mse_boot: dict | None = None
    mse_jkn: dict | None = None
    uid: int = int(dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")) + int(
        1e16 * rand.random()
    )

    @property
    def rse():
        pass

    @property
    def cv():
        pass

    def to_numpy():
        pass

    def to_polars():
        pass

    def to_pandas():
        pass


@frozen
class EbEst:
    area: list
    est: dict
    fit_stats: EbFit
    mse: dict | None = None
    mse_boot: dict | None = None
    mse_jkn: dict | None = None
    uid: int = int(dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")) + int(
        1e16 * rand.random()
    )

    @property
    def rse():
        pass

    @property
    def cv():
        pass

    def to_numpy():
        pass

    def to_polars():
        pass

    def to_pandas():
        pass
