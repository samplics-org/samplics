import numpy as np
import pandas as pd
import pytest

from samplics.sampling import Sample
from samplics.utils import checks, formats

countries_population = pd.read_csv("./tests/sampling/countries_population_2019.csv")

countries = countries_population["country"]
population_size = countries_population["population_2019"]

sample_size = int(np.rint(countries.size / 40))


"""Simple random sampling"""
grs_design = Sample(method="grs")


def test_grs_select():
    unnormalized_probs = np.random.sample(countries.size) * countries.size
    grs_probs = unnormalized_probs / np.sum(unnormalized_probs)
    grs_sample, grs_number_hits = grs_design.select(countries, sample_size, probs=grs_probs)
    assert sum(grs_sample) <= sample_size
    assert sum(grs_number_hits) == sample_size


srs_design_wr = Sample(method="srs")


def test_srswr_probs():
    srs_probs = srs_design_wr.inclusion_probs(countries, sample_size)
    return np.isclose(srs_probs.all(), sample_size / countries.size)


def test_srswr_select():
    srs_sample, srs_number_hits = srs_design_wr.select(countries, sample_size)
    assert np.sum(srs_sample) <= sample_size
    assert np.sum(srs_number_hits) == sample_size


srs_design_wor = Sample(method="srs", with_replacement=False)


def test_srswor_probs():
    srs_probs = srs_design_wor.inclusion_probs(countries, sample_size)
    return np.isclose(srs_probs.all(), sample_size / countries.size)


def test_srswor_select():
    srs_sample, srs_number_hits = srs_design_wor.select(countries, sample_size)
    assert np.sum(srs_sample) == sample_size
    assert np.sum(srs_number_hits) == sample_size
    assert (np.unique(srs_number_hits) == (0, 1)).all()


"""Stratified simple random sampling"""
continent = countries_population["continent"]
sample_sizes = dict(
    {"Africa": 10, "Asia": 7, "Europe": 5, "North America": 3, "South America": 3, "Oceanie": 1,}
)

str_srswr_design = Sample(method="srs", stratification=True)


def test_stratified_srswr_probs():
    str_srswr_probs = str_srswr_design.inclusion_probs(countries, sample_sizes, continent)
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_srswr_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_srswr_select():
    str_srswr_sample, str_srswr_number_hits = str_srswr_design.select(
        countries, sample_sizes, continent
    )
    strata = np.unique(continent)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_srswr_number_hits[continent == s])
    assert sample_sizes == obtained_sample_sizes


str_srswor_design = Sample(method="srs", stratification=True, with_replacement=False)


def test_stratified_srswor_probs():
    str_srswor_probs = str_srswor_design.inclusion_probs(countries, sample_sizes, continent)
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_srswor_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_srswor_select():
    str_srswor_sample, str_srswor_number_hits = str_srswor_design.select(
        countries, sample_sizes, continent
    )
    strata = np.unique(continent)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_srswor_number_hits[continent == s])
    assert (np.unique(str_srswor_number_hits) == (0, 1)).all()
    assert sample_sizes == obtained_sample_sizes


"""PPS sampling"""
mos = countries_population["population_2019"].to_numpy()

pps_sys_design_wr = Sample(method="pps-sys")


def test_ppswr_sys_probs():
    pps_probs = pps_sys_design_wr.inclusion_probs(countries, sample_size, mos=mos)
    return np.isclose(pps_probs.all(), sample_size * mos / np.sum(mos))


def test_ppswr_sys_select():
    pps_sample, pps_number_hits = pps_sys_design_wr.select(countries, sample_size, mos=mos)
    assert np.sum(pps_sample) <= sample_size
    assert np.sum(pps_number_hits) == sample_size


pps_hv_design_wor = Sample(method="pps-hv", with_replacement=False)


def test_ppswor_hv_probs():
    pps_probs = pps_hv_design_wor.inclusion_probs(countries, sample_size, mos=mos)
    return np.isclose(pps_probs.all(), sample_size * mos / np.sum(mos))


def test_ppswor_hv_select():
    pps_sample, pps_number_hits = pps_hv_design_wor.select(countries, sample_size, mos=mos)
    assert np.sum(pps_sample) == sample_size
    assert np.sum(pps_number_hits) == sample_size
    assert (np.unique(pps_number_hits) == (0, 1)).all()


pps_brewer_design_wor = Sample(method="pps-brewer", with_replacement=False)


def test_ppswor_brewer_probs():
    pps_probs = pps_brewer_design_wor.inclusion_probs(countries, sample_size, mos=mos)
    return np.isclose(pps_probs.all(), sample_size * mos / np.sum(mos))


def test_ppswor_brewer_select():
    pps_sample, pps_number_hits = pps_brewer_design_wor.select(countries, sample_size, mos=mos)
    assert np.sum(pps_sample) == sample_size
    assert np.sum(pps_number_hits) == sample_size
    assert (np.unique(pps_number_hits) == (0, 1)).all()


pps_murphy_design_wor = Sample(method="pps-murphy", with_replacement=False)


def test_ppswor_murphy_probs():
    pps_probs = pps_murphy_design_wor.inclusion_probs(countries, 2, mos=mos)
    return np.isclose(pps_probs.all(), sample_size * mos / np.sum(mos))


def test_ppswor_murphy_select():
    pps_sample, pps_number_hits = pps_murphy_design_wor.select(countries, 2, mos=mos)
    assert np.sum(pps_sample) == 2
    assert np.sum(pps_number_hits) == 2
    assert (np.unique(pps_number_hits) == (0, 1)).all()


# """Stratified PPS sampling"""
mos[mos > np.mean(mos)] = mos[mos > np.mean(mos)] * 0.1
sample_sizes_pps = dict(
    {"Africa": 5, "Asia": 4, "Europe": 3, "North America": 2, "South America": 2, "Oceanie": 1,}
)

str_ppswr_sys_design = Sample(method="pps-sys", stratification=True)


def test_stratified_ppswr_sys_probs():
    str_ppswr_sys_probs = str_ppswr_sys_design.inclusion_probs(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_ppswr_sys_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_ppswr_sys_select():
    str_ppswr_sys_sample, str_ppswr_sys_number_hits = str_ppswr_sys_design.select(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_ppswr_sys_number_hits[continent == s])
    assert sample_sizes_pps == obtained_sample_sizes


str_ppswr_hv_design = Sample(method="pps-hv", stratification=True)


def test_stratified_ppswr_hv_probs():
    str_ppswr_hv_probs = str_ppswr_hv_design.inclusion_probs(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_ppswr_hv_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_ppswr_hv_select():
    str_ppswr_hv_sample, str_ppswr_hv_number_hits = str_ppswr_hv_design.select(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_ppswr_hv_number_hits[continent == s])
    assert sample_sizes_pps == obtained_sample_sizes


str_ppswr_brewer_design = Sample(method="pps-brewer", stratification=True)


def test_stratified_ppswr_brewer_probs():
    str_ppswr_brewer_probs = str_ppswr_brewer_design.inclusion_probs(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_ppswr_brewer_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_ppswr_brewer_select():
    (str_ppswr_brewer_sample, str_ppswr_brewer_number_hits,) = str_ppswr_brewer_design.select(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_ppswr_brewer_number_hits[continent == s])
    assert sample_sizes_pps == obtained_sample_sizes


countries_population_murphy = countries_population.loc[
    countries_population["continent"] != "Oceanie"
]

countries_murphy = countries_population_murphy["country"]
mos_murphy = countries_population_murphy["population_2019"].to_numpy()
mos_murphy[mos_murphy > np.mean(mos_murphy)] = mos_murphy[mos_murphy > np.mean(mos_murphy)] * 0.1
sample_size_murphy = int(np.rint(countries_murphy.size / 40))
continent_murphy = countries_population_murphy["continent"]
sample_sizes_murphy = dict(
    {"Africa": 2, "Asia": 2, "Europe": 2, "North America": 2, "South America": 2,}
)

str_ppswr_murphy_design = Sample(method="pps-murphy", stratification=True)


def test_stratified_ppswr_murphy_probs():
    str_ppswr_murphy_probs = str_ppswr_murphy_design.inclusion_probs(
        countries_murphy, sample_sizes_murphy, continent_murphy, mos=mos_murphy,
    )
    strata = np.unique(continent_murphy)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent_murphy == s)
        obtained_probs[s] = np.unique(str_ppswr_murphy_probs[continent_murphy == s])
    assert initial_probs == obtained_probs


def test_stratified_ppswr_murphy_select():
    (str_ppswr_murphy_sample, str_ppswr_murphy_number_hits,) = str_ppswr_murphy_design.select(
        countries_murphy, sample_sizes_murphy, continent_murphy, mos=mos_murphy,
    )
    strata = np.unique(continent_murphy)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_ppswr_murphy_number_hits[continent_murphy == s])
    assert sample_sizes_murphy == obtained_sample_sizes
