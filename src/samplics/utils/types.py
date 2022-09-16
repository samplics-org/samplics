"""Provides the custom types used throughout the modules.
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, Union

import numpy as np
import pandas as pd


Array = Union[np.ndarray, pd.Series, list, tuple]
Series = Union[pd.Series, list, tuple]

Number = Union[float, int]
StringNumber = Union[str, float, int]

DictStrNum = Dict[StringNumber, Number]
DictStrInt = Dict[StringNumber, int]
DictStrFloat = Dict[StringNumber, float]
DictStrBool = Dict[StringNumber, bool]

# Population parameters
@unique
class PopParam(Enum):
    mean = "Mean"
    total = "Total"
    prop = "Proportion"


# Methods for sample size
@unique
class SizeMethod(Enum):
    wald = "Wald"
    fleiss = "Fleiss"


@unique
class SelectMethod(Enum):
    srs = "SRS"
    sys = "Systematic"
    pps_brewer = "PPS-Brewer"
    pps_hv = "PPS-HanuravVijayan"  # Hanurav-Vijayan
    pps_murphy = "PPS-Murphy"
    pps_rs = "PPS-RaoSampford"  # Rao-Sampford
    pps_sys = "PPS-Systematic"
    grs = "General"


@unique
class SinglePSUEst(Enum):
    """Estimation options for strata with singleton PSU"""

    error = "Raise Error when one PSU in a stratum"
    skip = "Set variance to zero and skip stratum with one PSU"
    certainty = "Use SSUs or lowest units to estimate the variance"
    combine = "Combine the strata with the singleton psu to another stratum"


@unique
class SamplicsError(Enum):
    """Type of errors"""

    SinglePSU = "Only one PSU in the stratum"
