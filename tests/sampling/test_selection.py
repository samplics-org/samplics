import numpy as np
import pandas as pd

from samplics.sampling import SampleSelection
from samplics.utils.types import SelectMethod


countries_population = pd.read_csv("./tests/sampling/countries_population_2019.csv")

countries = countries_population["country"]
population_size = countries_population["population_2019"]

sample_size = int(np.rint(countries.size / 40))

np.random.seed(12345)


"""Simple random sampling"""
grs_design = SampleSelection(method=SelectMethod.grs)


def test_grs_select():
    unnormalized_probs = np.random.sample(countries.size) * countries.size
    probs = unnormalized_probs / np.sum(unnormalized_probs)
    grs_sample, grs_number_hits, grs_probs = grs_design.select(
        samp_unit=countries, samp_size=sample_size, probs=probs
    )
    assert sum(grs_sample) <= sample_size
    assert sum(grs_number_hits) == sample_size
    assert (grs_probs == probs).all()


srs_design_wr = SampleSelection(method=SelectMethod.srs_wr)


def test_srswr_probs():
    srs_probs = srs_design_wr._inclusion_probs(countries, sample_size)
    assert (np.isclose(srs_probs, sample_size / countries.size)).all()


def test_srswr_select():
    srs_sample, srs_number_hits, srs_probs = srs_design_wr.select(countries, sample_size)
    assert np.sum(srs_sample) <= sample_size
    assert np.sum(srs_number_hits) == sample_size
    assert (np.isclose(srs_probs, sample_size / countries.size)).all()


srs_design_wor = SampleSelection(method=SelectMethod.srs_wor)


def test_srswor_probs():
    srs_probs = srs_design_wor._inclusion_probs(countries, sample_size)
    assert (np.isclose(srs_probs, sample_size / countries.size)).all()


def test_srswor_select():
    srs_sample, srs_number_hits, _ = srs_design_wor.select(countries, sample_size)
    assert np.sum(srs_sample) == sample_size
    assert np.sum(srs_number_hits) == sample_size
    assert (np.unique(srs_number_hits) == (0, 1)).all()


"""Stratified simple random sampling"""
continent = countries_population["continent"]
sample_sizes = dict(
    {
        "Africa": 10,
        "Asia": 7,
        "Europe": 5,
        "North America": 3,
        "South America": 3,
        "Oceanie": 1,
    }
)

str_srswr_design = SampleSelection(method=SelectMethod.srs_wr, strat=True)


def test_stratified_srswr_probs_same_size():
    size = 2
    str_srswr_probs = str_srswr_design._inclusion_probs(countries, size, continent)
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = size / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_srswr_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_srswr_probs():
    str_srswr_probs = str_srswr_design._inclusion_probs(countries, sample_sizes, continent)
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_srswr_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_srswr_select_same_size():
    size = 2
    str_srswr_sample, str_srswr_number_hits, _ = str_srswr_design.select(
        samp_unit=countries, samp_size=size, stratum=continent
    )
    strata = np.unique(continent)
    # obtained_sample_sizes = dict()
    for s in strata:
        assert size == np.sum(str_srswr_number_hits[continent == s])


def test_stratified_srswr_select():
    str_srswr_sample, str_srswr_number_hits, _ = str_srswr_design.select(
        countries, sample_sizes, continent
    )
    strata = np.unique(continent)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_srswr_number_hits[continent == s])
    assert sample_sizes == obtained_sample_sizes


str_srswor_design = SampleSelection(method=SelectMethod.srs_wor, strat=True, wr=False)


def test_stratified_srswor_probs():
    str_srswor_probs = str_srswor_design._inclusion_probs(countries, sample_sizes, continent)
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_srswor_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_srswor_select_same_size():
    size = 2
    str_srswor_sample, str_srswor_number_hits, _ = str_srswor_design.select(
        countries, size, continent
    )
    strata = np.unique(continent)
    # obtained_sample_sizes = dict()
    for s in strata:
        assert size == np.sum(str_srswor_number_hits[continent == s])
    assert (np.unique(str_srswor_number_hits) == (0, 1)).all()


def test_stratified_srswor_select1():
    str_srswor_sample, str_srswor_number_hits, _ = str_srswor_design.select(
        countries, sample_sizes, continent
    )
    strata = np.unique(continent)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_srswor_number_hits[continent == s])
    assert (np.unique(str_srswor_number_hits) == (0, 1)).all()
    assert sample_sizes == obtained_sample_sizes


def test_stratified_srswor_select2():
    size = 2
    str_srswor_sample, str_srswor_number_hits, _ = str_srswor_design.select(
        countries, size, continent
    )
    strata = np.unique(continent)
    # obtained_sample_sizes = dict()
    for s in strata:
        assert size == np.sum(str_srswor_number_hits[continent == s])
    assert (np.unique(str_srswor_number_hits) == (0, 1)).all()


"""SYS sampling"""
str_syswor_design = SampleSelection(method=SelectMethod.sys, strat=True, wr=False)


def test_stratified_syswor_select():
    str_srswor_sample, str_srswor_number_hits, _ = str_syswor_design.select(
        countries, stratum=continent, samp_rate=0.3
    )


"""PPS sampling"""
mos = countries_population["population_2019"].to_numpy()

pps_sys_design_wr = SampleSelection(method=SelectMethod.pps_sys)


def test_ppswr_sys_probs():
    pps_probs = pps_sys_design_wr._inclusion_probs(countries, sample_size, mos=mos)
    assert (np.isclose(pps_probs, sample_size * mos / np.sum(mos))).all()


def test_ppswr_sys_select_same_size():
    size = 2
    pps_sample, pps_number_hits, pps_probs = pps_sys_design_wr.select(
        samp_unit=countries, samp_size=size, mos=mos
    )
    assert np.sum(pps_sample) <= size
    assert np.sum(pps_number_hits) == size
    assert np.isclose(pps_probs, size * mos / np.sum(mos)).all()


def test_ppswr_sys_select():
    pps_sample, pps_number_hits, pps_probs = pps_sys_design_wr.select(
        countries, sample_size, mos=mos
    )
    assert np.sum(pps_sample) <= sample_size
    assert np.sum(pps_number_hits) == sample_size
    assert np.isclose(pps_probs, sample_size * mos / np.sum(mos)).all()


def test_ppswr_sys_select_shuffle():
    pps_sample_shuffled, pps_number_hits_shuffled, pps_probs_shuffled = pps_sys_design_wr.select(
        countries, sample_size, mos=mos, shuffle=True
    )
    assert np.sum(pps_sample_shuffled) <= sample_size
    assert np.sum(pps_number_hits_shuffled) == sample_size
    # assert np.isclose(pps_probs_shuffled, sample_size * mos / np.sum(mos)).all()


pps_hv_design_wor = SampleSelection(method=SelectMethod.pps_hv, wr=False)


def test_ppswor_hv_probs():
    pps_probs = pps_hv_design_wor._inclusion_probs(countries, sample_size, mos=mos)
    assert (np.isclose(pps_probs, sample_size * mos / np.sum(mos))).all()


def test_ppswor_hv_select():
    pps_sample, pps_number_hits, _ = pps_hv_design_wor.select(countries, sample_size, mos=mos)
    assert np.sum(pps_sample) == sample_size
    assert np.sum(pps_number_hits) == sample_size
    assert (np.unique(pps_number_hits) == (0, 1)).all()


pps_brewer_design_wor = SampleSelection(method=SelectMethod.pps_brewer, wr=False)


def test_ppswor_brewer_probs():
    pps_probs = pps_brewer_design_wor._inclusion_probs(countries, sample_size, mos=mos)
    assert (np.isclose(pps_probs, sample_size * mos / np.sum(mos))).all()


def test_ppswor_brewer_select():
    pps_sample, pps_number_hits, _ = pps_brewer_design_wor.select(countries, sample_size, mos=mos)
    assert np.sum(pps_sample) == sample_size
    assert np.sum(pps_number_hits) == sample_size
    assert (np.unique(pps_number_hits) == (0, 1)).all()


pps_murphy_design_wor = SampleSelection(method=SelectMethod.pps_murphy, wr=False)


def test_ppswor_murphy_probs():
    pps_probs = pps_murphy_design_wor._inclusion_probs(countries, 2, mos=mos)
    assert (np.isclose(pps_probs, 2 * mos / np.sum(mos))).all()


def test_ppswor_murphy_select():
    pps_sample, pps_number_hits, _ = pps_murphy_design_wor.select(countries, 2, mos=mos)
    assert np.sum(pps_sample) == 2
    assert np.sum(pps_number_hits) == 2
    assert (np.unique(pps_number_hits) == (0, 1)).all()


# """Stratified PPS sampling"""
mos[mos > np.mean(mos)] = mos[mos > np.mean(mos)] * 0.1
sample_sizes_pps = dict(
    {
        "Africa": 5,
        "Asia": 4,
        "Europe": 3,
        "North America": 2,
        "South America": 2,
        "Oceanie": 1,
    }
)

str_ppswr_sys_design = SampleSelection(method=SelectMethod.pps_sys, strat=True)


def test_stratified_ppswr_sys_probs():
    str_ppswr_sys_probs = str_ppswr_sys_design._inclusion_probs(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_ppswr_sys_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_ppswr_sys_select():
    (
        str_ppswr_sys_sample,
        str_ppswr_sys_number_hits,
        _,
    ) = str_ppswr_sys_design.select(countries, sample_sizes_pps, continent, mos=mos)
    strata = np.unique(continent)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_ppswr_sys_number_hits[continent == s])
    assert sample_sizes_pps == obtained_sample_sizes


str_ppswr_hv_design = SampleSelection(method=SelectMethod.pps_hv, strat=True)


def test_stratified_ppswr_hv_probs():
    str_ppswr_hv_probs = str_ppswr_hv_design._inclusion_probs(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_ppswr_hv_probs[continent == s])
    assert initial_probs == obtained_probs


# def test_stratified_ppswr_hv_select():
#     (
#         str_ppswr_hv_sample,
#         str_ppswr_hv_number_hits,
#         _,
#     ) = str_ppswr_hv_design.select(countries, sample_sizes_pps, continent, mos=mos)
#     strata = np.unique(continent)
#     obtained_sample_sizes = dict()
#     for s in strata:
#         obtained_sample_sizes[s] = np.sum(str_ppswr_hv_number_hits[continent == s])
#     assert sample_sizes_pps == obtained_sample_sizes


str_ppswr_brewer_design = SampleSelection(method=SelectMethod.pps_brewer, strat=True)


def test_stratified_ppswr_brewer_probs():
    str_ppswr_brewer_probs = str_ppswr_brewer_design._inclusion_probs(
        countries, sample_sizes_pps, continent, mos=mos
    )
    strata = np.unique(continent)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent == s)
        obtained_probs[s] = np.unique(str_ppswr_brewer_probs[continent == s])
    assert initial_probs == obtained_probs


def test_stratified_ppswr_brewer_select():
    (
        str_ppswr_brewer_sample,
        str_ppswr_brewer_number_hits,
        _,
    ) = str_ppswr_brewer_design.select(countries, sample_sizes_pps, continent, mos=mos)
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
    {
        "Africa": 2,
        "Asia": 2,
        "Europe": 2,
        "North America": 2,
        "South America": 2,
    }
)

str_ppswr_murphy_design = SampleSelection(method=SelectMethod.pps_murphy, strat=True)


def test_stratified_ppswr_murphy_probs():
    str_ppswr_murphy_probs = str_ppswr_murphy_design._inclusion_probs(
        countries_murphy, sample_sizes_murphy, continent_murphy, mos=mos_murphy
    )
    strata = np.unique(continent_murphy)
    initial_probs = obtained_probs = dict()
    for s in strata:
        initial_probs[s] = sample_sizes[s] / np.sum(continent_murphy == s)
        obtained_probs[s] = np.unique(str_ppswr_murphy_probs[continent_murphy == s])
    assert initial_probs == obtained_probs


def test_stratified_ppswr_murphy_select():
    (str_ppswr_murphy_sample, str_ppswr_murphy_number_hits, _,) = str_ppswr_murphy_design.select(
        countries_murphy, sample_sizes_murphy, continent_murphy, mos=mos_murphy
    )
    strata = np.unique(continent_murphy)
    obtained_sample_sizes = dict()
    for s in strata:
        obtained_sample_sizes[s] = np.sum(str_ppswr_murphy_number_hits[continent_murphy == s])
    assert sample_sizes_murphy == obtained_sample_sizes
