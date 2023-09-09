import datetime as dt
import random as rand

from enum import Enum, unique

import numpy as np
import pandas as pd
import polars as pl

from attr import validators

from samplics.utils.formats import numpy_array
from samplics.utils.types import Array, DictStrNum, Number


# NumpyArray = TypeVar("np.ndarray", bound=np.ndarray)
# PandasDF = TypeVar("pd.DataFrame", bound=pd.DataFrame)
# PolarsDF = TypeVar("pl.DataFrame", bound=pl.DataFrame)


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


from attrs import field, frozen


@frozen
class DirectEst:
    area: tuple = field(validator=validators.instance_of(tuple))
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

        if not isinstance(area, tuple):
            area = tuple(numpy_array(area).flatten())

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

        self.__attrs_init__(area, est, stderr, ssize, psize)

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
                "area": list(self.area),
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
    area: tuple
    auxdata: dict
    record_id: tuple | None
    uid: int = int(
        dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        + str(int(1e16 * rand.random()))
    )

    def __init__(self, area: Array, record_id: Array, **kwargs) -> None:
        varnames = tuple(kwargs.keys())
        areas = numpy_array(area)
        areas_unique = tuple(np.unique(areas))
        record_id = numpy_array(record_id) if record_id is not None else None

        auxdata = {}
        for d in areas_unique:
            auxdata[d] = {}
            if record_id is not None:
                auxdata[d]["record_id"] = tuple(record_id[areas == area])
            else:
                auxdata[d]["record_id"] = None
        for varname in varnames:
            vardata = numpy_array(kwargs[varname])
            for d in areas:
                auxdata[d][varname] = tuple(vardata[areas == d])

        self.__attrs_init__(areas_unique, auxdata, record_id)

    def to_numpy(self, varlist: str | list[str] | None = None):
        return self.to_polars(varlist).to_numpy()

    def to_polars(self, varlist: str | list[str] | None = None):
        for i, d in enumerate(self.area):
            if i == 0:
                df = pl.from_dict(self.auxdata[d])
                df = df.insert_at_idx(1, pl.repeat(d, n=df.shape[0], eager=True).alias("area"))
            else:
                dfi = pl.from_dict(self.auxdata[d])
                df = pl.concat(
                    [
                        df,
                        dfi.insert_at_idx(
                            1, pl.repeat(d, n=dfi.shape[0], eager=True).alias("area")
                        ),
                    ]
                )

        return df if varlist is None else df.select(varlist)

    def to_pandas(self, varlist: str | list[str] | None = None):
        return self.to_polars(varlist).to_pandas()
