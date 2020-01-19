import numpy as np
import pandas as pd

from samplics.weighting import SampleWeight

import pytest

income_sample = pd.read_csv("./tests/weighting/synthetic_income_data.csv")


unit_id = income_sample["unit_id"]
cluster_id = income_sample["cluster_id"]
region_id = income_sample["region_id"]

response_status = income_sample["response_status"]


response_code = np.zeros(response_status.size).astype(int)
response_code[response_status == "RR"] = 1
response_code[response_status == "NR"] = 2
response_code[response_status == "UK"] = 3

response_map = dict({"in": 0, "rr": 1, "nr": 2, "uk": 3})


design_wgt = income_sample["design_wgt"]

sample_wgt = SampleWeight()

"""Non-response adjustment WITHOUT adjustment classes"""

nr_wgt_without_adj_class = sample_wgt.adjust(
    design_wgt, None, response_status=response_code, response_dict=response_map
)


def test_sum_of_weights_without_adj_class():
    assert np.sum(nr_wgt_without_adj_class) == np.sum(design_wgt)


def test_nr_adjustment_without_adj_class():
    nonrespondents = response_code == 2
    assert (nr_wgt_without_adj_class[nonrespondents] == 0).all()


def test_in_adjustment_without_adj_class():
    ineligibles = response_code == 0
    assert (nr_wgt_without_adj_class[ineligibles] == design_wgt[ineligibles]).all()


def test_uk_adjustment_without_adj_class():
    unknowns = response_code == 3
    assert (nr_wgt_without_adj_class[unknowns] == 0).all()


def test_deff_wgt_without_domain():
    mean_wgt = np.mean(nr_wgt_without_adj_class)
    deff_wgt = 1 + np.mean(
        np.power(nr_wgt_without_adj_class - mean_wgt, 2) / mean_wgt ** 2
    )
    assert sample_wgt.deff_weight(nr_wgt_without_adj_class)["__none__"] == deff_wgt
    assert sample_wgt.deff_wgt["__none__"] == deff_wgt


"""Non-response adjustment WITH adjustment classes"""

nr_wgt_with_adj_class = sample_wgt.adjust(
    design_wgt, region_id, response_code, response_map
)


def test_sum_of_weights_with_adj_class():
    assert np.sum(nr_wgt_with_adj_class) == np.sum(design_wgt)


def test_nr_adjustment_with_adj_class():
    nonrespondents = response_code == 2
    assert (nr_wgt_with_adj_class[nonrespondents] == 0).all()


def test_in_adjustment_with_adj_class():
    ineligibles = response_code == 0
    assert (nr_wgt_with_adj_class[ineligibles] == design_wgt[ineligibles]).all()


def test_uk_adjustment_with_adj_class():
    unknowns = response_code == 3
    assert (nr_wgt_with_adj_class[unknowns] == 0).all()


def test_deff_wgt_with_domain():
    deff_wgt_region = sample_wgt.deff_weight(nr_wgt_with_adj_class, region_id)
    for region in np.unique(region_id):
        nr_wgt_r = nr_wgt_with_adj_class[region_id == region]
        mean_wgt_r = np.mean(nr_wgt_r)
        deff_wgt_r = 1 + np.mean(np.power(nr_wgt_r - mean_wgt_r, 2) / mean_wgt_r ** 2)
        assert deff_wgt_region[region] == deff_wgt_r
        assert sample_wgt.deff_wgt[region] == deff_wgt_r


"""Normalization adjustment WITHOUT normalization class"""
level1_without = 50
norm_wgt_wihout_class1 = sample_wgt.normalize(
    nr_wgt_without_adj_class, norm_level=level1_without
)


def test_norm_wgt_without_class1():
    assert np.isclose(np.sum(norm_wgt_wihout_class1), level1_without)


norm_wgt_wihout_class2 = sample_wgt.normalize(nr_wgt_without_adj_class)


def test_norm_wgt_without_class2():
    assert np.isclose(np.sum(norm_wgt_wihout_class2), norm_wgt_wihout_class2.size)


"""Normalization adjustment WITH normalization class"""
region_ids = np.unique(region_id)
level1_with = dict(zip(region_ids, np.repeat(500, region_ids.size)))
norm_wgt_wih_class1 = sample_wgt.normalize(
    nr_wgt_without_adj_class, region_id, level1_with
)


def test_norm_wgt_with_class1():
    for region in region_ids:
        norm_wgt_wih_class_r = norm_wgt_wih_class1[region_id == region]
        response_code_r = response_code[region_id == region]
        respondents_r = response_code_r == 1
        assert np.isclose(np.sum(norm_wgt_wih_class_r), level1_with[region])


norm_wgt_wih_class2 = sample_wgt.normalize(nr_wgt_without_adj_class, region_id)


def test_norm_wgt_with_class2():
    for region in region_id:
        norm_wgt_wih_class_r = norm_wgt_wih_class2[region_id == region]
        response_code_r = response_code[region_id == region]
        assert np.isclose(np.sum(norm_wgt_wih_class_r), response_code_r.size)
