from samplics.apis.base.fitting import _fit
from samplics.types.containers import Array, DirectEst, FitMethod


def fit(y: DirectEst | Array, x, method):
    match y:
        case DirectEst():
            y_vec = y.est
            return _fit(y=y_vec, x=x, method=FitMethod.fh)
        case Array():
            return _fit(y=y, x=x, method=FitMethod.fh)
        case _:
            raise TypeError("The type of `y` is not supported!")
