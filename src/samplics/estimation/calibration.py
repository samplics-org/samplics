from __future__ import annotations

from typing import Optional, Union

# import numpy as np
# import pandas as pd

from samplics.utils.types import Array, Number, Series, StringNumber


class CalibrateEstimator:
    """Provides calibration based estimation"""

    def __init__(self, parameter: str, alpha: float = 0.05, random_seed: Optional[int] = None):
        pass

    def estimate(
        self,
        y: Array,
        calib_weight: Array,
        # calib_rep_weight: Optional[Array] = None,
        x: Optional[Array] = None,
        stratum: Optional[Series] = None,
        psu: Optional[Series] = None,
        ssu: Optional[Series] = None,
        domain: Optional[Series] = None,
        by: Optional[Series] = None,
        fpc: Union[dict[StringNumber, Number], Series, Number] = 1.0,
        deff: bool = False,
        coef_variation: bool = False,
        as_factor: bool = False,
        remove_nan: bool = False,
    ) -> None:
        pass