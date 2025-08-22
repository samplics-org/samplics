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


def test_srswr_select():
    srs_sample, srs_number_hits, srs_probs = srs_design_wr.select(countries, sample_size)
    assert np.sum(srs_sample) <= sample_size
    assert np.sum(srs_number_hits) == sample_size
    assert (np.isclose(srs_probs, sample_size / countries.size)).all()


srs_design_wor = SampleSelection(method=SelectMethod.srs_wor)


def test_srswor_select():
    srs_sample, srs_number_hits, srs_probs = srs_design_wor.select(countries, sample_size)
    assert np.sum(srs_sample) == sample_size
    assert np.sum(srs_number_hits) == sample_size
    assert (np.unique(srs_number_hits) == (0, 1)).all()
    assert (np.isclose(srs_probs, sample_size / countries.size)).all()


def test_srswor_select_to_dataframet():
    sample_df = srs_design_wor.select(
        samp_unit=countries, samp_size=sample_size, to_dataframe=True
    )
    assert (sample_df.columns == ["_samp_unit", "_sample", "_hits", "_probs"]).all()


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


def test_stratified_srswr_select_same_size():
    size = 2
    str_srswr_sample, str_srswr_number_hits, str_srswr_probs = str_srswr_design.select(
        samp_unit=countries, samp_size=size, stratum=continent
    )
    strata = np.unique(continent)
    # obtained_sample_sizes = dict()
    for s in strata:
        assert size == np.sum(str_srswr_number_hits[continent == s])
        assert size / np.sum(continent == s) == np.unique(str_srswr_probs[continent == s])


def test_stratified_srswr_select():
    str_srswr_sample, str_srswr_number_hits, str_srswr_probs = str_srswr_design.select(
        countries, sample_sizes, continent
    )
    strata = np.unique(continent)
    for s in strata:
        assert sample_sizes[s] == np.sum(str_srswr_number_hits[continent == s])
        assert sample_sizes[s] / np.sum(continent == s) == np.unique(
            str_srswr_probs[continent == s]
        )


str_srswor_design = SampleSelection(method=SelectMethod.srs_wor, strat=True, wr=False)


def test_stratified_srswor_select_same_size():
    size = 2
    (
        str_srswor_sample,
        str_srswor_number_hits,
        str_srswor_probs,
    ) = str_srswor_design.select(countries, size, continent)
    strata = np.unique(continent)
    for s in strata:
        assert size == np.sum(str_srswor_number_hits[continent == s])
        assert size / np.sum(continent == s) == np.unique(str_srswor_probs[continent == s])
    assert (np.unique(str_srswor_number_hits) == (0, 1)).all()


def test_stratified_srswor_select1():
    (
        str_srswor_sample,
        str_srswor_number_hits,
        str_srswor_probs,
    ) = str_srswor_design.select(countries, sample_sizes, continent)
    strata = np.unique(continent)
    for s in strata:
        assert sample_sizes[s] == np.sum(str_srswor_number_hits[continent == s])
        assert sample_sizes[s] / np.sum(continent == s) == np.unique(
            str_srswor_probs[continent == s]
        )
    assert (np.unique(str_srswor_number_hits) == (0, 1)).all()


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


def test_stratified_syswor_select2():
    str_srswor_sample, str_srswor_number_hits, _ = str_syswor_design.select(
        countries, stratum=continent, samp_size=sample_sizes
    )


"""PPS sampling"""
mos = countries_population["population_2019"].to_numpy()

pps_sys_design_wr = SampleSelection(method=SelectMethod.pps_sys)


def test_ppswr_sys_select_same_size():
    size = 2
    pps_sample, pps_hits, pps_probs = pps_sys_design_wr.select(
        samp_unit=countries, samp_size=size, mos=mos
    )
    assert np.sum(pps_sample) <= size
    assert np.sum(pps_hits) == size
    assert np.isclose(pps_probs, size * mos / np.sum(mos)).all()


def test_ppswr_sys_select():
    pps_sample, pps_hits, pps_probs = pps_sys_design_wr.select(countries, sample_size, mos=mos)
    assert np.sum(pps_sample) <= sample_size
    assert np.sum(pps_hits) == sample_size
    assert np.isclose(pps_probs, sample_size * mos / np.sum(mos)).all()


def test_ppswr_sys_select_shuffle():
    (
        pps_sample_shuffled,
        pps_hits_shuffled,
        pps_probs_shuffled,
    ) = pps_sys_design_wr.select(countries, sample_size, mos=mos, shuffle=True)
    assert np.sum(pps_sample_shuffled) <= sample_size
    assert np.sum(pps_hits_shuffled) == sample_size
    # assert np.isclose(pps_probs_shuffled, sample_size * mos / np.sum(mos)).all()


pps_hv_design_wor = SampleSelection(method=SelectMethod.pps_hv, wr=False)


def test_ppswor_hv_select():
    pps_sample, pps_hits, pps_probs = pps_hv_design_wor.select(countries, sample_size, mos=mos)
    assert np.sum(pps_sample) == sample_size
    assert np.sum(pps_hits) == sample_size
    assert (np.unique(pps_hits) == (0, 1)).all()
    assert (np.isclose(pps_probs, sample_size * mos / np.sum(mos))).all()


pps_brewer_design_wor = SampleSelection(method=SelectMethod.pps_brewer, wr=False)


def test_ppswor_brewer_select():
    pps_sample, pps_hits, pps_probs = pps_brewer_design_wor.select(countries, sample_size, mos=mos)
    assert np.sum(pps_sample) == sample_size
    assert np.sum(pps_hits) == sample_size
    assert (np.unique(pps_hits) == (0, 1)).all()
    assert (np.isclose(pps_probs, sample_size * mos / np.sum(mos))).all()


pps_murphy_design_wor = SampleSelection(method=SelectMethod.pps_murphy, wr=False)


def test_ppswor_murphy_select():
    pps_sample, pps_hits, pps_probs = pps_murphy_design_wor.select(countries, 2, mos=mos)
    assert np.sum(pps_sample) == 2
    assert np.sum(pps_hits) == 2
    assert (np.unique(pps_hits) == (0, 1)).all()
    assert (np.isclose(pps_probs, 2 * mos / np.sum(mos))).all()


pps_rs_design_wor = SampleSelection(method=SelectMethod.pps_rs, wr=False)


def test_ppswor_rs_select():
    pps_sample, pps_hits, pps_probs = pps_rs_design_wor.select(
        samp_unit=countries, samp_size=2, mos=mos
    )
    assert np.sum(pps_sample) == 2
    assert np.sum(pps_hits) == 2
    assert (np.unique(pps_hits) == (0, 1)).all()
    assert (np.isclose(pps_probs, 2 * mos / np.sum(mos))).all()


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


def test_stratified_ppswr_sys_select():
    (
        str_ppswr_sys_sample,
        str_ppswr_sys_number_hits,
        str_ppswr_sys_probs,
    ) = str_ppswr_sys_design.select(countries, sample_sizes_pps, continent, mos=mos)
    strata = np.unique(continent)
    for s in strata:
        assert sample_sizes_pps[s] == np.sum(str_ppswr_sys_number_hits[continent == s])
        assert (
            sample_sizes_pps[s] * mos[continent == s] / np.sum(mos[continent == s])
            == str_ppswr_sys_probs[continent == s]
        ).all()


str_ppswr_hv_design = SampleSelection(method=SelectMethod.pps_hv, strat=True)


def test_stratified_ppswr_hv_select():
    (
        str_ppswr_hv_sample,
        str_ppswr_hv_number_hits,
        str_ppswr_hv_probs,
    ) = str_ppswr_hv_design.select(countries, sample_sizes_pps, continent, mos=mos)
    strata = np.unique(continent)
    for s in strata:
        assert sample_sizes_pps[s] == np.sum(str_ppswr_hv_number_hits[continent == s])
        assert (
            sample_sizes_pps[s] * mos[continent == s] / np.sum(mos[continent == s])
            == str_ppswr_hv_probs[continent == s]
        ).all()


str_ppswr_brewer_design = SampleSelection(method=SelectMethod.pps_brewer, strat=True)


def test_stratified_ppswr_brewer_select():
    (
        str_ppswr_brewer_sample,
        str_ppswr_brewer_number_hits,
        str_ppswr_brewer_probs,
    ) = str_ppswr_brewer_design.select(countries, sample_sizes_pps, continent, mos=mos)
    strata = np.unique(continent)
    for s in strata:
        assert sample_sizes_pps[s] == np.sum(str_ppswr_brewer_number_hits[continent == s])
        assert (
            sample_sizes_pps[s] * mos[continent == s] / np.sum(mos[continent == s])
            == str_ppswr_brewer_probs[continent == s]
        ).all()


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


def test_stratified_ppswr_murphy_select():
    (
        str_ppswr_murphy_sample,
        str_ppswr_murphy_number_hits,
        str_ppswr_murphy_probs,
    ) = str_ppswr_murphy_design.select(
        countries_murphy, sample_sizes_murphy, continent_murphy, mos=mos_murphy
    )
    strata = np.unique(continent_murphy)
    for s in strata:
        assert sample_sizes_murphy[s] == np.sum(
            str_ppswr_murphy_number_hits[continent_murphy == s]
        )
        assert (
            sample_sizes_murphy[s]
            * mos_murphy[continent_murphy == s]
            / np.sum(mos_murphy[continent_murphy == s])
            == str_ppswr_murphy_probs[continent_murphy == s]
        ).all()


def test_stratified_ppswr_to_dataframet():
    sample_df = str_ppswr_murphy_design.select(
        samp_unit=countries_murphy,
        samp_size=sample_sizes_murphy,
        stratum=continent_murphy,
        mos=mos_murphy,
        to_dataframe=True,
    )
    assert (
        sample_df.columns == ["_samp_unit", "_stratum", "_mos", "_sample", "_hits", "_probs"]
    ).all()


# Private methods


def test__calculate_samp_size_int():
    some_design = SampleSelection(method=SelectMethod.pps_murphy, strat=False)
    samp_size = some_design._calculate_samp_size(strat=False, pop_size=100, samp_rate=0.1)
    assert samp_size == 10


def test__calculate_samp_size_dict():
    some_design = SampleSelection(method=SelectMethod.pps_murphy, strat=True)
    samp_size = some_design._calculate_samp_size(
        strat=True,
        pop_size={"one": 100, "two": 200},
        samp_rate={"one": 0.1, "two": 0.20},
    )
    assert samp_size["one"] == 10
    assert samp_size["two"] == 40


def test__calculate_samp_rate_int():
    some_design = SampleSelection(method=SelectMethod.pps_murphy, strat=False)
    samp_rate = some_design._calculate_samp_rate(strat=False, pop_size=100, samp_size=35)
    assert samp_rate == 35 / 100


def test__calculate_samp_rate_dict():
    some_design = SampleSelection(method=SelectMethod.pps_murphy, strat=True)
    samp_rate = some_design._calculate_samp_rate(
        strat=True, pop_size={"one": 100, "two": 200}, samp_size={"one": 10, "two": 20}
    )
    assert samp_rate["one"] == 10 / 100
    assert samp_rate["two"] == 20 / 200


def test__calculate_fpc_int():
    some_design = SampleSelection(method=SelectMethod.pps_murphy, strat=False)

    fpc = some_design._calculate_fpc(strat=False, pop_size=100, samp_size=35)
    assert fpc == np.sqrt((100 - 35) / (100 - 1))


def test__calculate_fpc_dict():
    some_design = SampleSelection(method=SelectMethod.pps_murphy, strat=True)
    fpc = some_design._calculate_fpc(
        strat=True, pop_size={"one": 100, "two": 200}, samp_size={"one": 10, "two": 20}
    )
    assert fpc["one"] == np.sqrt((100 - 10) / (100 - 1))
    assert fpc["two"] == np.sqrt((200 - 20) / (200 - 1))
