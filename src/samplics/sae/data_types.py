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
from samplics.utils.types import DF, Array, DictStrNum, Number


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

    def to_numpy(self, varlist: str | list[str] | None = None):
        return self.to_polars(varlist).to_numpy()

    def to_polars(self, varlist: str | list[str] | None = None):
        aux_df = pl.from_dict(
            {
                "areas": list(self.areas),
                "est": list(self.est.values()),
                "stderr": list(self.stderr.values()),
                "ssize": list(self.ssize.values()),
            }
        )

        if varlist is None:
            return aux_df
            # return aux_df.select(["est", "stderr", "ssize"]).to_numpy()
        elif isinstance(varlist, (str, list)):
            return aux_df.select(varlist)
        else:
            raise TypeError("varlist must be None or str or list[str]")

    def to_pandas(self, varlist: str | list[str] | None = None):
        return self.to_polars(varlist).to_pandas()


@frozen
class AuxVars:
    areas: list
    auxdata: dict
    # ssize: dict
    # record_id: tuple | None
    uid: int = int(
        dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        + str(int(1e16 * rand.random()))
    )

    def __init__(
        self,
        area: Array,
        auxdata: DF | Array | Iterable[DF | Array] | None = None,
        record_id: Array = None,
        **kwargs,
    ) -> None:
        assert isinstance(area, Array)

        area = numpy_array(area)
        assert area.shape[0] > 0

        areas_unique = np.unique(area).tolist()
        record_id = (
            numpy_array(record_id)
            if record_id is not None
            else np.linspace(0, area.shape[0] - 1, area.shape[0]).astype(int)
        )

        if isinstance(auxdata, (DF, Array)):
            aux_df = self.__from_df(auxdata)
            if isinstance(auxdata, Array):
                aux_df.columns = ["__aux_" + str(i) for i in range(aux_df.shape[1])]
        elif isinstance(auxdata, Iterable):
            for i, d in enumerate(auxdata):
                assert isinstance(d, (DF, Array))
                if i == 0:
                    aux_df = self.__from_df(d)
                    if isinstance(d, Array):
                        aux_df.columns = ["__aux_" + str(i) for i in range(aux_df.shape[1])]
                else:
                    d_df = self.__from_df(d)
                    if isinstance(d, Array):
                        d_df.columns = ["__aux_" + str(i) for i in range(d_df.shape[1])]
                    aux_df.hstack([d_df], in_place=True)
        else:
            aux_df = None

        if aux_df is None:
            auxdata = pl.from_dict(kwargs)
        else:
            auxdata = aux_df.hstack(pl.from_dict(kwargs))

        auxdata_dict = (
            auxdata.insert_at_idx(0, pl.from_numpy(area).to_series().alias("area"))
            .insert_at_idx(0, pl.Series(record_id).alias("record_id"))
            .partition_by("area", as_dict=True)
        )

        auxdata_dict = {k: auxdata_dict[k].to_dict(as_series=False) for k in auxdata_dict}

        # ssize = {}
        for k in auxdata_dict:
            # ssize[k] = len(auxdata_dict[k]["area"])
            del auxdata_dict[k]["area"]

        # self.__attrs_init__(areas_unique, auxdata_dict, ssize)
        self.__attrs_init__(areas_unique, auxdata_dict)

    def __from_df(self, auxdata: DF | Array) -> pl.DataFrame | None:
        if isinstance(auxdata, pl.DataFrame):
            return auxdata
        elif isinstance(auxdata, pl.Series):
            return pl.DataFrame(auxdata)
        elif isinstance(auxdata, (pd.DataFrame, pd.Series)):
            return pl.from_pandas(auxdata)
        elif isinstance(auxdata, np.ndarray):
            return pl.from_numpy(auxdata)
        elif isinstance(auxdata, Array):
            return pl.DataFrame(auxdata)
        else:
            return None

    def to_numpy(self, varlist: str | list[str] | None = None):
        return self.to_polars(varlist).to_numpy()

    def to_polars(self, varlist: str | list[str] | None = None):
        return pl.concat(
            [
                pl.from_dict(self.auxdata[d]).insert_at_idx(
                    1, pl.repeat(d, n=self.ssize[d], eager=True).alias("area")
                )
                for d in self.area
            ]
        )

    def to_pandas(self, varlist: str | list[str] | None = None):
        return self.to_polars(varlist).to_pandas()


class CovMat:
    pass


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
