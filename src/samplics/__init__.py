# from samplics.apis import Frame, Sample
from samplics.categorical import CrossTabulation, Tabulation, Ttest
from samplics.datasets.datasets import (
    load_auto,
    load_birth,
    load_county_crop,
    load_county_crop_means,
    load_expenditure_milk,
    load_nhanes2,
    load_nhanes2brr,
    load_nhanes2jk,
    load_nmihs,
    load_psu_frame,
    load_psu_sample,
    load_ssu_sample,
)
from samplics.estimation import ReplicateEstimator, TaylorEstimator
from samplics.regression import SurveyGLM
from samplics.sae import (
    EblupAreaModel,
    EblupUnitModel,
    EbUnitModel,
    EllUnitModel,
)
from samplics.sampling import (
    SampleSelection,
    SampleSize,
    SampleSizeMeanOneSample,
    SampleSizeMeanTwoSample,
    SampleSizePropOneSample,
    SampleSizePropTwoSample,
    allocate,
    calculate_power,
    calculate_power_prop,
    calculate_ss_fleiss_prop,
    calculate_ss_wald_mean,
    calculate_ss_wald_prop,
    power_for_one_mean,
    power_for_one_proportion,
)

# from samplics.types import (
#     AuxVars,
#     DirectEst,
#     EbEst,
#     EbFit,
#     EblupEst,
#     EblupFit,
#     FitMethod,
#     FitStats,
#     Mse,
# )
from samplics.utils import (
    CertaintyError,
    DimensionError,
    FitMethod,
    MethodError,
    PopParam,
    ProbError,
    RepMethod,
    SamplicsError,
    SelectMethod,
    SinglePSUError,
    SinglePSUEst,
    SizeMethod,
    array_to_dict,
    transform,
)
from samplics.weighting import ReplicateWeight, SampleWeight


# From pkgs


__pkgs__ = []


__all__ = __pkgs__ + [
    "allocate",
    "array_to_dict",
    # "AuxVars",
    "calculate_power",
    "calculate_power_prop",
    "calculate_ss_fleiss_prop",
    "calculate_ss_wald_prop",
    "calculate_ss_wald_mean",
    "CrossTabulation",
    # "DirectEst",
    "Tabulation",
    "Ttest",
    "EblupAreaModel",
    "EblupUnitModel",
    # "EblupEst",
    # "EblupFit",
    "EbUnitModel",
    # "EbEst",
    # "EbFit",
    "EllUnitModel",
    "FitMethod",
    "fit_eblup",
    # "FitStats",
    "load_auto",
    "load_birth",
    "load_county_crop",
    "load_county_crop_means",
    "load_expenditure_milk",
    "load_nhanes2",
    "load_nhanes2brr",
    "load_nhanes2jk",
    "load_nmihs",
    "load_psu_frame",
    "load_psu_sample",
    "load_ssu_sample",
    "PopParam",
    "power_for_one_mean",
    "power_for_one_proportion",
    "SampleSelection",
    "SampleSize",
    "SampleSizeMeanOneSample",
    "SampleSizeMeanTwoSample",
    "SampleSizePropOneSample",
    "SampleSizePropTwoSample",
    "SampleWeight",
    "SelectMethod",
    "SinglePSUEst",
    "SizeMethod",
    "SurveyGLM",
    "ReplicateWeight",
    "RepMethod",
    "ReplicateEstimator",
    "TaylorEstimator",
    "transform",
    # Custom exception classes
    "SamplicsError",
    "CertaintyError",
    "DimensionError",
    "MethodError",
    # "Mse",
    "ProbError",
    "SinglePSUError",
    # Objects from APIs sub-package
    "Frame",
    "Sample",
]

__version__ = "0.4.24"
