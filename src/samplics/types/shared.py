import datetime as dt
import random as rand

from collections.abc import Iterable
from enum import Enum, unique
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl

from attrs import frozen

from samplics.utils.formats import numpy_array
from samplics.utils.types import DF, Array


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


@frozen
class AuxVars:
    domains: list
    auxdata: dict
    nrecords: dict
    uid: int = int(
        dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        + str(int(1e16 * rand.random()))
    )

    def __init__(
        self,
        domain: Array,
        auxdata: DF | Array | Iterable[DF | Array] | None = None,
        record_id: Array = None,
        **kwargs,
    ) -> None:
        assert isinstance(domain, Array)

        domain = numpy_array(domain)
        assert domain.shape[0] > 0

        record_id = (
            numpy_array(record_id)
            if record_id is not None
            else np.linspace(0, domain.shape[0] - 1, domain.shape[0]).astype(int)
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
                        aux_df.columns = [
                            "__aux_" + str(i) for i in range(aux_df.shape[1])
                        ]
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
            auxdata.insert_at_idx(0, pl.from_numpy(domain).to_series().alias("area"))
            .insert_at_idx(0, pl.Series(record_id).alias("record_id"))
            .partition_by("area", as_dict=True)
        )

        auxdata_dict = {
            k: auxdata_dict[k].to_dict(as_series=False) for k in auxdata_dict
        }
        nrecords = {k: len(auxdata_dict[k]["area"]) for k in auxdata_dict}

        for k in auxdata_dict:
            del auxdata_dict[k]["area"]

        self.__attrs_init__(np.unique(domain).tolist(), auxdata_dict, nrecords)

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
        auxdata = pl.concat(
            [
                pl.from_dict(self.auxdata[d]).insert_at_idx(
                    1, pl.repeat(d, n=self.nrecords[d], eager=True).alias("area")
                )
                for d in self.domains
            ]
        )

        if keep_vars is None and drop_vars is None:
            varlist = auxdata.columns
        elif keep_vars is not None:
            varlist = keep_vars
        elif drop_vars is not None:
            varlist = [item for item in auxdata.columns if item not in drop_vars]
        else:
            pass

        return auxdata.select(varlist)

    def to_pandas(
        self,
        keep_vars: str | Iterable[str] | None = None,
        drop_vars: str | Iterable[str] | None = None,
    ):
        return self.to_polars(keep_vars=keep_vars, drop_vars=drop_vars).to_pandas()


class CovMat:
    pass
