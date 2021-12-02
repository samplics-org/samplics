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
from samplics.sae import EblupAreaModel, EblupUnitModel, EbUnitModel, EllUnitModel
from samplics.sampling import (
    SampleSelection,
    SampleSize,
    SampleSizeOneMean,
    SampleSizeOneProportion,
    SampleSizeOneTotal,
    allocate,
    calculate_power,
    power_for_one_mean,
    power_for_one_proportion,
    power_for_proportion,
    sample_size_for_mean_wald,
    sample_size_for_proportion_fleiss,
    sample_size_for_proportion_wald,
)
from samplics.utils.basic_functions import transform
from samplics.utils.formats import array_to_dict
from samplics.weighting import ReplicateWeight, SampleWeight


__all__ = [
    "allocate",
    "array_to_dict",
    "calculate_power",
    "CrossTabulation",
    "Tabulation",
    "Ttest",
    "EblupAreaModel",
    "EblupUnitModel",
    "EbUnitModel",
    "EllUnitModel",
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
    "power_for_one_mean",
    "power_for_proportion",
    "power_for_one_proportion",
    "SampleSelection",
    "SampleSize",
    "SampleSizeOneMean",
    "SampleSizeOneProportion",
    "SampleSizeOneTotal",
    "sample_size_for_mean_wald",
    "sample_size_for_proportion_fleiss",
    "sample_size_for_proportion_wald",
    "SampleWeight",
    "SurveyGLM",
    "ReplicateWeight",
    "ReplicateEstimator",
    "TaylorEstimator",
    "transform",
]

__version__ = "0.3.13"
