import numpy as np
import polars as pl

from samplics.weighting import SampleWeight


# stata example

# nhis_sam = pl.read_csv("~/Downloads/nhis_sam.csv").with_columns(
#     pl.when(pl.col("hisp") == 4).then(pl.lit(3)).otherwise(pl.col("hisp")).alias("hisp")
# )

# age_grp = {
#     "<18": 5991,
#     "18-24": 2014,
#     "25-44": 6124,
#     "45-64": 5011,
#     "65+": 2448,
# }
# hisp_race = {1: 5031, 2: 12637, 3: 3920}
# control = {"age_grp": age_grp, "hisp": hisp_race}

# # breakpoint()

# ll = 0.8
# ul = 1.2

# margins = {
#     "age_grp": nhis_sam["age_grp"].to_list(),
#     "hisp": nhis_sam["hisp"].to_list(),
# }

# nhis_sam_rk = SampleWeight()

# nhis_sam = nhis_sam.with_columns(
#     rake_wt_2=nhis_sam_rk.rake(
#         samp_weight=nhis_sam["wt"], control=control, margins=margins, display_iter=True, tol=1e-6
#     )
# ).with_columns(diff=pl.col("rake_wt_2") - pl.col("rake_wt"))

# synthetic data for testing

wgt = np.random.uniform(0, 1, 1000)
resp = np.random.choice([0, 1], p=(0.3, 0.7), size=1000)


def test_nr_adjust_incomplete_classes1():
    resp_map = dict({"nr": 0, "rr": 1})
    sample_synth = SampleWeight()
    samp_synth_wgt = sample_synth.adjust(
        samp_weight=wgt,
        adj_class=None,
        resp_status=resp,
        resp_dict=resp_map,
        unknown_to_inelig=False,
    )

    assert np.isclose(np.sum(samp_synth_wgt[resp == 1]), np.sum(wgt))
    assert np.isclose(np.sum(samp_synth_wgt[resp == 0]), 0)


def test_nr_adjust_incomplete_classes2():
    resp_map = dict({"nr": 0, "rr": 1, "in": 2})
    sample_synth = SampleWeight()
    samp_synth_wgt = sample_synth.adjust(
        samp_weight=wgt,
        adj_class=None,
        resp_status=resp,
        resp_dict=resp_map,
        unknown_to_inelig=False,
    )

    assert np.isclose(np.sum(samp_synth_wgt[resp == 1]), np.sum(wgt))
    assert np.isclose(np.sum(samp_synth_wgt[resp == 0]), 0)


def test_nr_adjust_incomplete_classes3():
    resp_map = dict({"nr": 0, "rr": 1, "in": 2})
    resp = np.random.choice([0, 1, 2], p=(0.2, 0.7, 0.1), size=1000)
    sample_synth = SampleWeight()
    samp_synth_wgt = sample_synth.adjust(
        samp_weight=wgt,
        adj_class=None,
        resp_status=resp,
        resp_dict=resp_map,
        unknown_to_inelig=True,
    )

    assert np.isclose(
        np.sum(samp_synth_wgt[resp == 1]) + np.sum(samp_synth_wgt[resp == 2]),
        np.sum(wgt),
    )
    assert np.isclose(np.sum(samp_synth_wgt[resp == 0]), 0)


def test_nr_adjust_complete_classes():
    resp_map = dict({"nr": 0, "rr": 1, "in": 2, "uk": 3})
    resp = np.random.choice([0, 1, 2, 3], p=(0.2, 0.5, 0.1, 0.2), size=1000)
    sample_synth = SampleWeight()
    samp_synth_wgt = sample_synth.adjust(
        samp_weight=wgt,
        adj_class=None,
        resp_status=resp,
        resp_dict=resp_map,
        unknown_to_inelig=True,
    )

    assert np.isclose(
        np.sum(samp_synth_wgt[resp == 1]) + np.sum(samp_synth_wgt[resp == 2]),
        np.sum(wgt),
    )
    assert np.isclose(np.sum(samp_synth_wgt[resp == 0]), 0)
    assert np.isclose(np.sum(samp_synth_wgt[resp == 3]), 0)


# real data for testing
income_sample = pl.read_csv("./tests/weighting/synthetic_income_data.csv")


unit_id = income_sample["unit_id"]
cluster_id = income_sample["cluster_id"]
region_id = income_sample["region_id"]

response_status = income_sample["response_status"]
# print(pd.crosstab(response_status, response_status))

response_code = np.zeros(response_status.shape[0]).astype(int)
response_code[response_status == "RR"] = 1  # np.repeat(1, 4091)
response_code[response_status == "NR"] = 2  # np.repeat(2, 709)
response_code[response_status == "UK"] = 3  # np.repeat(3, 149)

response_map = dict({"in": 0, "rr": 1, "nr": 2, "uk": 3})

design_wgt = income_sample["design_wgt"]


# """Non-response adjustment WITHOUT adjustment classes"""
sample_wgt_nr_without = SampleWeight()

nr_wgt_without_adj_class = sample_wgt_nr_without.adjust(
    samp_weight=design_wgt,
    adj_class=None,
    resp_status=response_code,
    resp_dict=response_map,
    unknown_to_inelig=False,
)


def test_nr_adj_method():
    assert sample_wgt_nr_without.adj_method == "nonresponse"


def test_sum_of_weights_without_adj_class():
    assert np.sum(nr_wgt_without_adj_class) == design_wgt.sum()


def test_nr_adjustment_without_adj_class():
    nonrespondents = response_code == np.repeat(2, response_code.size)
    assert (nr_wgt_without_adj_class[nonrespondents] == 0).all()


def test_in_adjustment_without_adj_class():
    ineligibles = response_code == np.repeat(0, response_code.size)
    assert (
        nr_wgt_without_adj_class[ineligibles] == design_wgt.to_numpy()[ineligibles]
    ).all()


def test_uk_adjustment_without_adj_class():
    unknowns = response_code == np.repeat(3, response_code.size)
    assert (nr_wgt_without_adj_class[unknowns] == 0).all()


def test_deff_weight_without_domain():
    mean_wgt = np.mean(nr_wgt_without_adj_class)
    deff_weight = 1 + np.mean(
        np.power(nr_wgt_without_adj_class - mean_wgt, 2) / mean_wgt**2
    )
    assert (
        sample_wgt_nr_without.calculate_deff_weight(nr_wgt_without_adj_class)
        == deff_weight
    )
    assert sample_wgt_nr_without.deff_weight == deff_weight


"""Non-response adjustment WITH adjustment classes"""
sample_wgt_nr_with = SampleWeight()

nr_wgt_with_adj_class = sample_wgt_nr_with.adjust(
    design_wgt, region_id, response_code, response_map, unknown_to_inelig=False
)


def test_sum_of_weights_with_adj_class():
    assert np.sum(nr_wgt_with_adj_class) == design_wgt.sum()


def test_nr_adjustment_with_adj_class():
    nonrespondents = response_code == np.repeat(2, response_code.size)
    assert (nr_wgt_with_adj_class[nonrespondents] == 0).all()


def test_in_adjustment_with_adj_class():
    ineligibles = response_code == np.repeat(0, response_code.size)
    assert (
        nr_wgt_with_adj_class[ineligibles] == design_wgt.to_numpy()[ineligibles]
    ).all()


def test_uk_adjustment_with_adj_class():
    unknowns = response_code == np.repeat(3, response_code.size)
    assert (nr_wgt_with_adj_class[unknowns] == 0).all()


def test_deff_weight_with_domain():
    deff_weight_region = sample_wgt_nr_with.calculate_deff_weight(
        nr_wgt_with_adj_class, region_id
    )
    for region in np.unique(region_id):
        nr_wgt_r = nr_wgt_with_adj_class[region_id == region]
        mean_wgt_r = np.mean(nr_wgt_r)
        deff_weight_r = 1 + np.mean(np.power(nr_wgt_r - mean_wgt_r, 2) / mean_wgt_r**2)
        assert (deff_weight_region[region] == deff_weight_r).all()
        assert (sample_wgt_nr_with.deff_weight[region] == deff_weight_r).all()


"""Normalization adjustment WITHOUT normalization class"""
level1_without = 50
sample_wgt_norm_without = SampleWeight()

norm_wgt_wihout_class1 = sample_wgt_norm_without.normalize(
    nr_wgt_without_adj_class, control=level1_without
)


def test_norm_without_adj_method():
    assert sample_wgt_norm_without.adj_method == "normalization"


def test_norm_wgt_without_class1():
    assert np.isclose(np.sum(norm_wgt_wihout_class1), level1_without)


norm_wgt_wihout_class2 = sample_wgt_norm_without.normalize(nr_wgt_without_adj_class)


def test_norm_wgt_without_class2():
    assert np.isclose(np.sum(norm_wgt_wihout_class2), norm_wgt_wihout_class2.size)


"""Normalization adjustment WITH normalization class"""
region_ids = np.unique(region_id)
level1_with = dict(zip(region_ids, np.repeat(500, region_ids.size)))
sample_wgt_norm_with = SampleWeight()

norm_wgt_wih_class1 = sample_wgt_norm_with.normalize(
    nr_wgt_without_adj_class, level1_with, region_id
)


def test_norm_with_adj_method():
    assert sample_wgt_norm_with.adj_method == "normalization"


def test_norm_wgt_with_class1():
    for region in region_ids:
        norm_wgt_wih_class_r = norm_wgt_wih_class1[region_id == region]
        response_code[region_id == region]
        # respondents_r = response_code_r == np.ones(response_code_r.size)
        assert np.isclose(np.sum(norm_wgt_wih_class_r), level1_with[region])


norm_wgt_wih_class2 = sample_wgt_norm_with.normalize(
    nr_wgt_without_adj_class, domain=region_id
)


def test_norm_wgt_with_class2():
    for region in region_id:
        norm_wgt_wih_class_r = norm_wgt_wih_class2[region_id == region]
        response_code_r = response_code[region_id == region]
        assert np.isclose(np.sum(norm_wgt_wih_class_r), response_code_r.size)


"""Postratification adjustment WITHOUT poststratification class"""
control_without = 50000
sample_wgt_ps_without = SampleWeight()

ps_wgt_wihout_class = sample_wgt_ps_without.poststratify(
    nr_wgt_without_adj_class, control_without
)


def test_ps_without_adj_method():
    assert sample_wgt_ps_without.adj_method == "poststratification"


def test_poststratification_wgt_without_class():
    assert np.isclose(np.sum(ps_wgt_wihout_class), control_without)


"""Poststratification adjustment WITH poststratification class"""
region_ids = np.unique(region_id)
control_with = dict(zip(region_ids, np.repeat(5000, region_ids.size)))
sample_wgt_ps_with = SampleWeight()

ps_wgt_wih_class = sample_wgt_ps_with.poststratify(
    nr_wgt_without_adj_class, control_with, domain=region_id
)


def test_ps_with_adj_method():
    assert sample_wgt_ps_with.adj_method == "poststratification"


def test_ps_wgt_with_class():
    for region in region_ids:
        ps_wgt_wih_class_r = ps_wgt_wih_class[region_id == region]
        # response_code_r = response_code[region_id == region]
        # respondents_r = response_code_r == np.ones(response_code_r.size)
        assert np.isclose(np.sum(ps_wgt_wih_class_r), control_with[region])


"""Raking adjustment"""

educ_grp = {
    "0. Primary": 240_000,
    "1. High-School": 1_600_000,
    "2. University": 800_000,
}
inc_grp = {"low": 600_000, "middle": 1_400_000, "high": 640_000}
control = {"educ_level": educ_grp, "income_level": inc_grp}

income_sample2 = income_sample.filter(
    pl.col("educ_level").is_not_null(), pl.col("income_level").is_not_null()
)

margins = {
    "educ_level": income_sample2["educ_level"].to_list(),
    "income_level": income_sample2["income_level"].to_list(),
}

sample_wgt_rk_not_bound = SampleWeight()

# rk_wgt_not_bound = sample_wgt_rk_not_bound.rake(
#     samp_weight=income_sample2["design_wgt"], control=control, margins=margins, display_iter=True, tol=1e-4
# )


# breakpoint()

# age_grp = {"<18": 21588, age}
