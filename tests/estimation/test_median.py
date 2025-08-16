import polars as pl

from samplics.estimation import TaylorEstimator
from samplics.utils.types import PopParam


api_strat = pl.read_csv("./tests/regression/api_strat.csv")

y = api_strat["api00"]
stratum = api_strat["stype"]
psu = api_strat["dnum"]
weight = api_strat["pw"]


def test_median():
    svy_median = TaylorEstimator(param=PopParam.median)
    svy_median.estimate(y=y, samp_weight=weight, remove_nan=True)


def test_median_psu():
    svy_median = TaylorEstimator(param=PopParam.median)
    svy_median.estimate(y=y, samp_weight=weight, psu=psu, remove_nan=True)


def test_median_psu_strata():
    svy_median = TaylorEstimator(param=PopParam.median)
    svy_median.estimate(y=y, samp_weight=weight, psu=psu, stratum=stratum, remove_nan=True)


def test_median_psu_domain():
    svy_median = TaylorEstimator(param=PopParam.median)
    svy_median.estimate(y=y, samp_weight=weight, psu=psu, domain=stratum, remove_nan=True)
