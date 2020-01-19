import numpy as np
import pandas as pd

import math

from samplics.weighting import ReplicateWeight

from samplics.estimation import ReplicateEstimator

import pprint
import pytest

"""Jackknife estimates"""
nhanes2jkn = pd.read_csv("./tests/estimation/nhanes2jknife.csv")

y_jkn = nhanes2jkn["weight"].values
x_jkn = nhanes2jkn["height"].values
z_jkn = nhanes2jkn["race"].values  # proportion
female_jkn = nhanes2jkn["female"].values

sample_wgt_jkn = nhanes2jkn["finalwgt"].astype("float")
sample_wgt_jkn = sample_wgt_jkn.values
domain_jkn = nhanes2jkn["region"].values
rep_wgt_jkn = nhanes2jkn.loc[:, "jkw_1":"jkw_62"].astype("float")
rep_wgt_jkn = rep_wgt_jkn.values


def test_jkn_mean():
    jkn_mean = ReplicateEstimator("jackknife", "mean")
    jkn_mean.estimate(
        y_jkn, sample_wgt_jkn, rep_wgt_jkn, conservative=True, exclude_nan=True
    )
    jkn_var = jkn_mean.variance
    jkn_stderr = pow(jkn_var.get("__none__"), 0.5)
    assert np.isclose(jkn_stderr, 0.2320822, atol=1e-7)


def test_jkn_mean_d():
    jkn_mean_d = ReplicateEstimator("jackknife", "mean")
    jkn_mean_d.estimate(
        y_jkn,
        sample_wgt_jkn,
        rep_wgt_jkn,
        domain=domain_jkn,
        conservative=True,
        exclude_nan=True,
    )
    jkn_var_d = jkn_mean_d.variance
    jkn_stderr_d1 = pow(jkn_var_d.get(1), 0.5)
    jkn_stderr_d2 = pow(jkn_var_d.get(2), 0.5)
    jkn_stderr_d3 = pow(jkn_var_d.get(3), 0.5)
    jkn_stderr_d4 = pow(jkn_var_d.get(4), 0.5)
    assert np.isclose(jkn_stderr_d1, 0.5825663, atol=1e-7)
    assert np.isclose(jkn_stderr_d2, 0.3493011, atol=1e-7)
    assert np.isclose(jkn_stderr_d3, 0.4890297, atol=1e-7)
    assert np.isclose(jkn_stderr_d4, 0.4432099, atol=1e-7)


jkn_total = jkn_mean = ReplicateEstimator("jackknife", "total")


def test_jkn_total():
    jkn_total = ReplicateEstimator("jackknife", "total")
    jkn_total.estimate(
        female_jkn, sample_wgt_jkn, rep_wgt_jkn, conservative=True, exclude_nan=True
    )
    jkn_var = jkn_total.variance
    jkn_stderr = pow(jkn_var.get("__none__"), 0.5)
    assert np.isclose(jkn_stderr, 1958480.0, atol=1e-1)


def test_jkn_total_d():
    jkn_total_d = ReplicateEstimator("jackknife", "total")
    jkn_total_d.estimate(
        female_jkn,
        sample_wgt_jkn,
        rep_wgt_jkn,
        domain=domain_jkn,
        conservative=True,
        exclude_nan=True,
    )
    jkn_var_d = jkn_total_d.variance
    jkn_stderr_d1 = pow(jkn_var_d.get(1), 0.5)
    jkn_stderr_d2 = pow(jkn_var_d.get(2), 0.5)
    jkn_stderr_d3 = pow(jkn_var_d.get(3), 0.5)
    jkn_stderr_d4 = pow(jkn_var_d.get(4), 0.5)
    assert np.isclose(jkn_stderr_d1, 414764.8, atol=1e-1)
    assert np.isclose(jkn_stderr_d2, 718338.9, atol=1e-1)
    assert np.isclose(jkn_stderr_d3, 1241590.0, atol=1e-1)
    assert np.isclose(jkn_stderr_d4, 1267301.0, atol=1e-1)


def test_jkn_prop():
    jkn_prop = ReplicateEstimator("jackknife", "proportion")
    jkn_prop.estimate(
        z_jkn, sample_wgt_jkn, rep_wgt_jkn, conservative=True, exclude_nan=True
    )
    jkn_var = jkn_prop.variance
    jkn_stderr_1 = pow(jkn_var.get("__none__")[1], 0.5)
    jkn_stderr_2 = pow(jkn_var.get("__none__")[2], 0.5)
    jkn_stderr_3 = pow(jkn_var.get("__none__")[3], 0.5)
    assert np.isclose(jkn_stderr_1, 0.0234118, atol=1e-7)
    assert np.isclose(jkn_stderr_2, 0.0178846, atol=1e-7)
    assert np.isclose(jkn_stderr_3, 0.0147892, atol=1e-7)


def test_jkn_prop_d():
    jkn_prop_d = ReplicateEstimator("jackknife", "proportion")
    jkn_prop_d.estimate(
        z_jkn,
        sample_wgt_jkn,
        rep_wgt_jkn,
        domain=domain_jkn,
        conservative=True,
        exclude_nan=True,
    )
    jkn_var_d = jkn_prop_d.variance
    jkn_stderr_d1_1 = pow(jkn_var_d.get(1)[1], 0.5)
    jkn_stderr_d1_2 = pow(jkn_var_d.get(1)[2], 0.5)
    jkn_stderr_d1_3 = pow(jkn_var_d.get(1)[3], 0.5)
    jkn_stderr_d2_1 = pow(jkn_var_d.get(2)[1], 0.5)
    jkn_stderr_d2_2 = pow(jkn_var_d.get(2)[2], 0.5)
    jkn_stderr_d2_3 = pow(jkn_var_d.get(2)[3], 0.5)
    jkn_stderr_d3_1 = pow(jkn_var_d.get(3)[1], 0.5)
    jkn_stderr_d3_2 = pow(jkn_var_d.get(3)[2], 0.5)
    jkn_stderr_d3_3 = pow(jkn_var_d.get(3)[3], 0.5)
    jkn_stderr_d4_1 = pow(jkn_var_d.get(4)[1], 0.5)
    jkn_stderr_d4_2 = pow(jkn_var_d.get(4)[2], 0.5)
    jkn_stderr_d4_3 = pow(jkn_var_d.get(4)[3], 0.5)
    assert np.isclose(jkn_stderr_d1_1, 0.0160647, atol=1e-7)
    assert np.isclose(jkn_stderr_d1_2, 0.0151277, atol=1e-7)
    assert np.isclose(jkn_stderr_d1_3, 0.0032544, atol=1e-7)
    assert np.isclose(jkn_stderr_d2_1, 0.0273711, atol=1e-7)
    assert np.isclose(jkn_stderr_d2_2, 0.0253503, atol=1e-7)
    assert np.isclose(jkn_stderr_d2_3, 0.0038415, atol=1e-7)
    assert np.isclose(jkn_stderr_d3_1, 0.0555942, atol=1e-7)
    assert np.isclose(jkn_stderr_d3_2, 0.0540677, atol=1e-7)
    assert np.isclose(jkn_stderr_d3_3, 0.0027513, atol=1e-7)
    assert np.isclose(jkn_stderr_d4_1, 0.0610838, atol=1e-7)
    assert np.isclose(jkn_stderr_d4_2, 0.0335221, atol=1e-7)
    assert np.isclose(jkn_stderr_d4_3, 0.0514410, atol=1e-7)


def test_jkn_ratio():
    jkn_ratio = ReplicateEstimator("jackknife", "ratio")
    jkn_ratio.estimate(
        y_jkn, sample_wgt_jkn, rep_wgt_jkn, x=x_jkn, conservative=True, exclude_nan=True
    )
    jkn_var = jkn_ratio.variance
    jkn_stderr = pow(jkn_var.get("__none__"), 0.5)
    assert np.isclose(jkn_stderr, 0.0012466, atol=1e-7)


def test_jkn_ratio_d():
    jkn_ratio_d = ReplicateEstimator("jackknife", "ratio")
    jkn_ratio_d.estimate(
        y_jkn,
        sample_wgt_jkn,
        rep_wgt_jkn,
        x=x_jkn,
        domain=domain_jkn,
        conservative=True,
        exclude_nan=True,
    )
    jkn_var_d = jkn_ratio_d.variance
    jkn_stderr_d1 = pow(jkn_var_d.get(1), 0.5)
    jkn_stderr_d2 = pow(jkn_var_d.get(2), 0.5)
    jkn_stderr_d3 = pow(jkn_var_d.get(3), 0.5)
    jkn_stderr_d4 = pow(jkn_var_d.get(4), 0.5)
    assert np.isclose(jkn_stderr_d1, 0.0031952, atol=1e-7)
    assert np.isclose(jkn_stderr_d2, 0.0019580, atol=1e-7)
    assert np.isclose(jkn_stderr_d3, 0.0025715, atol=1e-7)
    assert np.isclose(jkn_stderr_d4, 0.0023108, atol=1e-7)


"""BRR estimates"""

nhanes2brr = pd.read_csv("./tests/estimation/nhanes2brr.csv")

y_brr = nhanes2brr["weight"].values
x_brr = nhanes2brr["height"].values
z_brr = nhanes2brr["diabetes"].values  # proportion
female_brr = nhanes2brr["female"].values

sample_wgt_brr = nhanes2brr["finalwgt"].astype("float")
sample_wgt_brr = sample_wgt_brr.values
domain_brr = nhanes2brr["region"].values
rep_wgt_brr = nhanes2brr.loc[:, "brr_1":"brr_32"].astype("float")
rep_wgt_brr = rep_wgt_brr.values


def test_brr_mean():
    brr_mean = ReplicateEstimator("brr", "mean")
    brr_mean.estimate(
        y_brr, sample_wgt_brr, rep_wgt_brr, conservative=True, exclude_nan=True
    )
    brr_var = brr_mean.variance
    brr_stderr = pow(brr_var.get("__none__"), 0.5)
    assert np.isclose(brr_stderr, 0.1656454, atol=1e-7)


def test_brr_mean_d():
    brr_mean_d = ReplicateEstimator("brr", "mean")
    brr_mean_d.estimate(
        y_brr,
        sample_wgt_brr,
        rep_wgt_brr,
        x=x_brr,
        domain=domain_brr,
        conservative=True,
        exclude_nan=True,
    )
    brr_var_d = brr_mean_d.variance
    brr_stderr_d1 = pow(brr_var_d.get(1), 0.5)
    brr_stderr_d2 = pow(brr_var_d.get(2), 0.5)
    brr_stderr_d3 = pow(brr_var_d.get(3), 0.5)
    brr_stderr_d4 = pow(brr_var_d.get(4), 0.5)
    assert np.isclose(brr_stderr_d1, 0.4185291, atol=1e-7)
    assert np.isclose(brr_stderr_d2, 0.2506872, atol=1e-7)
    assert np.isclose(brr_stderr_d3, 0.3459414, atol=1e-7)
    assert np.isclose(brr_stderr_d4, 0.3182729, atol=1e-7)


def test_brr_total():
    brr_total = ReplicateEstimator("brr", "total")
    brr_total.estimate(
        female_brr, sample_wgt_brr, rep_wgt_brr, conservative=True, exclude_nan=True
    )
    brr_var = brr_total.variance
    brr_stderr = pow(brr_var.get("__none__"), 0.5)
    assert np.isclose(brr_stderr, 1396159, atol=1e-1)


def test_brr_total_d():
    brr_total_d = ReplicateEstimator("brr", "total")
    brr_total_d.estimate(
        female_brr,
        sample_wgt_brr,
        rep_wgt_brr,
        x=x_brr,
        domain=domain_brr,
        conservative=True,
        exclude_nan=True,
    )
    brr_var_d = brr_total_d.variance
    brr_stderr_d1 = pow(brr_var_d.get(1), 0.5)
    brr_stderr_d2 = pow(brr_var_d.get(2), 0.5)
    brr_stderr_d3 = pow(brr_var_d.get(3), 0.5)
    brr_stderr_d4 = pow(brr_var_d.get(4), 0.5)
    assert np.isclose(brr_stderr_d1, 295677.2, atol=1e-1)
    assert np.isclose(brr_stderr_d2, 512088.9, atol=1e-1)
    assert np.isclose(brr_stderr_d3, 885104.3, atol=1e-1)
    assert np.isclose(brr_stderr_d4, 903432.9, atol=1e-1)


def test_brr_prop():
    brr_prop = ReplicateEstimator("brr", "proportion")
    brr_prop.estimate(
        z_brr, sample_wgt_brr, rep_wgt_brr, conservative=True, exclude_nan=True
    )
    brr_var = brr_prop.variance
    brr_stderr_0 = pow(brr_var.get("__none__")[0.0], 0.5)
    brr_stderr_1 = pow(brr_var.get("__none__")[1.0], 0.5)
    assert np.isclose(brr_stderr_0, 0.0018145, atol=1e-7)
    assert np.isclose(brr_stderr_1, 0.0018145, atol=1e-7)


def test_brr_prop_d():
    brr_prop_d = ReplicateEstimator("brr", "proportion")
    brr_prop_d.estimate(
        z_brr,
        sample_wgt_brr,
        rep_wgt_brr,
        domain=domain_brr,
        conservative=True,
        exclude_nan=True,
    )
    brr_var_d = brr_prop_d.variance
    brr_stderr_d1_0 = pow(brr_var_d.get(1)[0.0], 0.5)
    brr_stderr_d1_1 = pow(brr_var_d.get(1)[1.0], 0.5)
    brr_stderr_d2_0 = pow(brr_var_d.get(2)[0.0], 0.5)
    brr_stderr_d2_1 = pow(brr_var_d.get(2)[1.0], 0.5)
    brr_stderr_d3_0 = pow(brr_var_d.get(3)[0.0], 0.5)
    brr_stderr_d3_1 = pow(brr_var_d.get(3)[1.0], 0.5)
    brr_stderr_d4_0 = pow(brr_var_d.get(4)[0.0], 0.5)
    brr_stderr_d4_1 = pow(brr_var_d.get(4)[1.0], 0.5)
    assert np.isclose(brr_stderr_d1_0, 0.0027924, atol=1e-7)
    assert np.isclose(brr_stderr_d1_1, 0.0027924, atol=1e-7)
    assert np.isclose(brr_stderr_d2_0, 0.0026819, atol=1e-7)
    assert np.isclose(brr_stderr_d2_1, 0.0026819, atol=1e-7)
    assert np.isclose(brr_stderr_d3_0, 0.0049079, atol=1e-7)
    assert np.isclose(brr_stderr_d3_1, 0.0049079, atol=1e-7)
    assert np.isclose(brr_stderr_d4_0, 0.0035924, atol=1e-7)
    assert np.isclose(brr_stderr_d4_1, 0.0035924, atol=1e-7)


def test_brr_ratio():
    brr_ratio = ReplicateEstimator("brr", "ratio")
    brr_ratio.estimate(
        y_brr, sample_wgt_brr, rep_wgt_brr, x=x_brr, conservative=True, exclude_nan=True
    )
    brr_var = brr_ratio.variance
    brr_stderr = pow(brr_var.get("__none__"), 0.5)
    assert np.isclose(brr_stderr, 0.0008904, atol=1e-7)


def test_brr_ratio_d():
    brr_ratio_d = ReplicateEstimator("brr", "ratio")
    brr_ratio_d.estimate(
        y_brr,
        sample_wgt_brr,
        rep_wgt_brr,
        x=x_brr,
        domain=domain_brr,
        conservative=True,
        exclude_nan=True,
    )
    brr_var_d = brr_ratio_d.variance
    brr_stderr_d1 = pow(brr_var_d.get(1), 0.5)
    brr_stderr_d2 = pow(brr_var_d.get(2), 0.5)
    brr_stderr_d3 = pow(brr_var_d.get(3), 0.5)
    brr_stderr_d4 = pow(brr_var_d.get(4), 0.5)
    assert np.isclose(brr_stderr_d1, 0.0022865, atol=1e-7)
    assert np.isclose(brr_stderr_d2, 0.0014021, atol=1e-7)
    assert np.isclose(brr_stderr_d3, 0.0018216, atol=1e-7)
    assert np.isclose(brr_stderr_d4, 0.0016590, atol=1e-7)


"""BRR-FAY estimates"""
fay_coef = 0.3

nhanes2fay = pd.read_csv("./tests/estimation/nhanes2fay.csv")

y_fay = nhanes2fay["weight"].values
x_fay = nhanes2fay["height"].values
z_fay = nhanes2fay["diabetes"].values  # proportion
female_fay = nhanes2fay["female"].values

sample_wgt_fay = nhanes2fay["finalwgt"].astype("float")
sample_wgt_fay = sample_wgt_fay.values
domain_fay = nhanes2fay["region"].values
rep_wgt_fay = nhanes2fay.loc[:, "fay_1":"fay_32"].astype("float")
rep_wgt_fay = rep_wgt_fay.values


def test_fay_mean():
    fay_mean = ReplicateEstimator("brr", "mean", fay_coef=fay_coef)
    fay_mean.estimate(
        y_fay, sample_wgt_fay, rep_wgt_fay, conservative=True, exclude_nan=True
    )
    fay_var = fay_mean.variance
    fay_stderr = pow(fay_var.get("__none__"), 0.5)
    assert np.isclose(fay_stderr, 0.1655724, atol=1e-7)


def test_fay_mean_d():
    fay_mean_d = ReplicateEstimator("brr", "mean", fay_coef=fay_coef)
    fay_mean_d.estimate(
        y_fay,
        sample_wgt_fay,
        rep_wgt_fay,
        domain=domain_fay,
        conservative=True,
        exclude_nan=True,
    )
    fay_var_d = fay_mean_d.variance
    fay_stderr_d1 = pow(fay_var_d.get(1), 0.5)
    fay_stderr_d2 = pow(fay_var_d.get(2), 0.5)
    fay_stderr_d3 = pow(fay_var_d.get(3), 0.5)
    fay_stderr_d4 = pow(fay_var_d.get(4), 0.5)
    assert np.isclose(fay_stderr_d1, 0.4174973, atol=1e-7)
    assert np.isclose(fay_stderr_d2, 0.2501473, atol=1e-7)
    assert np.isclose(fay_stderr_d3, 0.3465335, atol=1e-7)
    assert np.isclose(fay_stderr_d4, 0.3173425, atol=1e-7)


def test_fay_total():
    fay_total = ReplicateEstimator("brr", "total", fay_coef=fay_coef)
    fay_total.estimate(
        female_fay, sample_wgt_fay, rep_wgt_fay, conservative=True, exclude_nan=True
    )
    fay_var = fay_total.variance
    fay_stderr = pow(fay_var.get("__none__"), 0.5)
    assert np.isclose(fay_stderr, 1396159, atol=1e-1)


def test_fay_total_d():
    fay_total_d = ReplicateEstimator("brr", "total", fay_coef=fay_coef)
    fay_total_d.estimate(
        female_fay,
        sample_wgt_fay,
        rep_wgt_fay,
        domain=domain_fay,
        conservative=True,
        exclude_nan=True,
    )
    fay_var_d = fay_total_d.variance
    fay_stderr_d1 = pow(fay_var_d.get(1), 0.5)
    fay_stderr_d2 = pow(fay_var_d.get(2), 0.5)
    fay_stderr_d3 = pow(fay_var_d.get(3), 0.5)
    fay_stderr_d4 = pow(fay_var_d.get(4), 0.5)
    assert np.isclose(fay_stderr_d1, 295677.2, atol=1e-1)
    assert np.isclose(fay_stderr_d2, 512088.8, atol=1e-1)
    assert np.isclose(fay_stderr_d3, 885104.4, atol=1e-1)
    assert np.isclose(fay_stderr_d4, 903432.8, atol=1e-1)


def test_fay_prop():
    fay_prop = ReplicateEstimator("brr", "proportion", fay_coef=fay_coef)
    fay_prop.estimate(
        z_fay, sample_wgt_fay, rep_wgt_fay, conservative=True, exclude_nan=True
    )
    fay_var = fay_prop.variance
    fay_stderr_0 = pow(fay_var.get("__none__")[0.0], 0.5)
    fay_stderr_1 = pow(fay_var.get("__none__")[1.0], 0.5)
    assert np.isclose(fay_stderr_0, 0.0018143, atol=1e-7)
    assert np.isclose(fay_stderr_1, 0.0018143, atol=1e-7)


def test_fay_prop_d():
    fay_prop_d = ReplicateEstimator("brr", "proportion", fay_coef=fay_coef)
    fay_prop_d.estimate(
        z_fay,
        sample_wgt_fay,
        rep_wgt_fay,
        domain=domain_fay,
        conservative=True,
        exclude_nan=True,
    )
    fay_var_d = fay_prop_d.variance
    fay_stderr_d1_0 = pow(fay_var_d.get(1)[0.0], 0.5)
    fay_stderr_d1_1 = pow(fay_var_d.get(1)[1.0], 0.5)
    fay_stderr_d2_0 = pow(fay_var_d.get(2)[0.0], 0.5)
    fay_stderr_d2_1 = pow(fay_var_d.get(2)[1.0], 0.5)
    fay_stderr_d3_0 = pow(fay_var_d.get(3)[0.0], 0.5)
    fay_stderr_d3_1 = pow(fay_var_d.get(3)[1.0], 0.5)
    fay_stderr_d4_0 = pow(fay_var_d.get(4)[0.0], 0.5)
    fay_stderr_d4_1 = pow(fay_var_d.get(4)[1.0], 0.5)
    assert np.isclose(fay_stderr_d1_0, 0.0027889, atol=1e-7)
    assert np.isclose(fay_stderr_d1_1, 0.0027889, atol=1e-7)
    assert np.isclose(fay_stderr_d2_0, 0.0026818, atol=1e-7)
    assert np.isclose(fay_stderr_d2_1, 0.0026818, atol=1e-7)
    assert np.isclose(fay_stderr_d3_0, 0.0049001, atol=1e-7)
    assert np.isclose(fay_stderr_d3_1, 0.0049001, atol=1e-7)
    assert np.isclose(fay_stderr_d4_0, 0.0035400, atol=1e-7)
    assert np.isclose(fay_stderr_d4_1, 0.0035400, atol=1e-7)


def test_fay_ratio():
    fay_ratio = ReplicateEstimator("brr", "ratio", fay_coef=fay_coef)
    fay_ratio.estimate(
        y_fay, sample_wgt_fay, rep_wgt_fay, x=x_fay, conservative=True, exclude_nan=True
    )
    fay_var = fay_ratio.variance
    fay_stderr = pow(fay_var.get("__none__"), 0.5)
    assert np.isclose(fay_stderr, 0.0008898, atol=1e-7)


def test_fay_ratio_d():
    fay_ratio_d = ReplicateEstimator("brr", "ratio", fay_coef=fay_coef)
    fay_ratio_d.estimate(
        y_fay,
        sample_wgt_fay,
        rep_wgt_fay,
        x=x_fay,
        domain=domain_fay,
        conservative=True,
        exclude_nan=True,
    )
    fay_var_d = fay_ratio_d.variance
    fay_stderr_d1 = pow(fay_var_d.get(1), 0.5)
    fay_stderr_d2 = pow(fay_var_d.get(2), 0.5)
    fay_stderr_d3 = pow(fay_var_d.get(3), 0.5)
    fay_stderr_d4 = pow(fay_var_d.get(4), 0.5)
    assert np.isclose(fay_stderr_d1, 0.0022835, atol=1e-7)
    assert np.isclose(fay_stderr_d2, 0.0014000, atol=1e-7)
    assert np.isclose(fay_stderr_d3, 0.0018239, atol=1e-7)
    assert np.isclose(fay_stderr_d4, 0.0016543, atol=1e-7)


"""Bootstrap estimates"""
nmihsboot = pd.read_csv("./tests/estimation/nmihs_bs.csv")

y_boot = nmihsboot["birthwgt"].values
x_boot = y_boot * 0.6 + 500
z_boot = nmihsboot["highbp"].values  # proportion
married_boot = nmihsboot["marital"].values

sample_wgt_boot = nmihsboot["finwgt"].astype("float")
sample_wgt_boot = sample_wgt_boot.values
domain_boot = nmihsboot["agegrp"].values
rep_wgt_boot = nmihsboot.loc[:, "bsrw1":"bsrw1000"].astype("float")
rep_wgt_boot = rep_wgt_boot.values


def test_boot_mean():
    boot_mean = ReplicateEstimator("bootstrap", "mean")
    boot_mean.estimate(
        y_boot, sample_wgt_boot, rep_wgt_boot, conservative=True, exclude_nan=True
    )
    boot_var = boot_mean.variance
    boot_stderr = pow(boot_var.get("__none__"), 0.5)
    assert np.isclose(boot_stderr, 6.530090, atol=1e-6)


def test_boot_mean_d():
    boot_mean_d = ReplicateEstimator("bootstrap", "mean")
    boot_mean_d.estimate(
        y_boot,
        sample_wgt_boot,
        rep_wgt_boot,
        domain=domain_boot,
        conservative=True,
        exclude_nan=True,
    )
    boot_var_d = boot_mean_d.variance
    boot_stderr_d1 = pow(boot_var_d.get(1), 0.5)
    boot_stderr_d2 = pow(boot_var_d.get(2), 0.5)
    boot_stderr_d3 = pow(boot_var_d.get(3), 0.5)
    boot_stderr_d4 = pow(boot_var_d.get(4), 0.5)
    boot_stderr_d5 = pow(boot_var_d.get(5), 0.5)
    assert np.isclose(boot_stderr_d1, 18.47222, atol=1e-5)
    assert np.isclose(boot_stderr_d2, 11.90187, atol=1e-5)
    assert np.isclose(boot_stderr_d3, 12.12145, atol=1e-5)
    assert np.isclose(boot_stderr_d4, 15.77978, atol=1e-5)
    assert np.isclose(boot_stderr_d5, 27.63810, atol=1e-5)


def test_boot_total():
    boot_total = ReplicateEstimator("bootstrap", "total")
    boot_total.estimate(
        married_boot, sample_wgt_boot, rep_wgt_boot, conservative=True, exclude_nan=True
    )
    boot_var = boot_total.variance
    boot_stderr = pow(boot_var.get("__none__"), 0.5)
    assert np.isclose(boot_stderr, 18938.38, atol=1e-2)


def test_boot_total_d():
    boot_total_d = ReplicateEstimator("bootstrap", "total")
    boot_total_d.estimate(
        married_boot,
        sample_wgt_boot,
        rep_wgt_boot,
        domain=domain_boot,
        conservative=True,
        exclude_nan=True,
    )
    boot_var_d = boot_total_d.variance
    boot_stderr_d1 = pow(boot_var_d.get(1), 0.5)
    boot_stderr_d2 = pow(boot_var_d.get(2), 0.5)
    boot_stderr_d3 = pow(boot_var_d.get(3), 0.5)
    boot_stderr_d4 = pow(boot_var_d.get(4), 0.5)
    boot_stderr_d5 = pow(boot_var_d.get(5), 0.5)
    assert np.isclose(boot_stderr_d1, 11679.59, atol=1e-2)
    assert np.isclose(boot_stderr_d2, 22400.91, atol=1e-2)
    assert np.isclose(boot_stderr_d3, 23877.00, atol=1e-2)
    assert np.isclose(boot_stderr_d4, 20471.85, atol=1e-2)
    assert np.isclose(boot_stderr_d5, 14273.63, atol=1e-2)


def test_boot_prop():
    boot_prop = ReplicateEstimator("bootstrap", "proportion")
    boot_prop.estimate(
        z_boot, sample_wgt_boot, rep_wgt_boot, conservative=True, exclude_nan=True
    )
    boot_var = boot_prop.variance
    boot_stderr_0 = pow(boot_var.get("__none__")[0.0], 0.5)
    boot_stderr_1 = pow(boot_var.get("__none__")[1.0], 0.5)
    assert np.isclose(boot_stderr_0, 0.0028165, atol=1e-7)
    assert np.isclose(boot_stderr_1, 0.0028165, atol=1e-7)


def test_boot_prop_d():
    boot_prop_d = ReplicateEstimator("bootstrap", "proportion")
    boot_prop_d.estimate(
        z_boot,
        sample_wgt_boot,
        rep_wgt_boot,
        domain=domain_boot,
        conservative=True,
        exclude_nan=True,
    )
    boot_var_d = boot_prop_d.variance
    boot_stderr_d1_0 = pow(boot_var_d.get(1)[0.0], 0.5)
    boot_stderr_d1_1 = pow(boot_var_d.get(1)[1.0], 0.5)
    boot_stderr_d2_0 = pow(boot_var_d.get(2)[0.0], 0.5)
    boot_stderr_d2_1 = pow(boot_var_d.get(2)[1.0], 0.5)
    boot_stderr_d3_0 = pow(boot_var_d.get(3)[0.0], 0.5)
    boot_stderr_d3_1 = pow(boot_var_d.get(3)[1.0], 0.5)
    boot_stderr_d4_0 = pow(boot_var_d.get(4)[0.0], 0.5)
    boot_stderr_d4_1 = pow(boot_var_d.get(4)[1.0], 0.5)
    boot_stderr_d5_0 = pow(boot_var_d.get(5)[0.0], 0.5)
    boot_stderr_d5_1 = pow(boot_var_d.get(5)[1.0], 0.5)
    assert np.isclose(boot_stderr_d1_0, 0.0096789, atol=1e-7)
    assert np.isclose(boot_stderr_d1_1, 0.0096789, atol=1e-7)
    assert np.isclose(boot_stderr_d2_0, 0.0060554, atol=1e-7)
    assert np.isclose(boot_stderr_d2_1, 0.0060554, atol=1e-7)
    assert np.isclose(boot_stderr_d3_0, 0.0047095, atol=1e-7)
    assert np.isclose(boot_stderr_d3_1, 0.0047095, atol=1e-7)
    assert np.isclose(boot_stderr_d4_0, 0.0053543, atol=1e-7)
    assert np.isclose(boot_stderr_d4_1, 0.0053543, atol=1e-7)
    assert np.isclose(boot_stderr_d5_0, 0.0096153, atol=1e-7)
    assert np.isclose(boot_stderr_d5_1, 0.0096153, atol=1e-7)


def test_boot_ratio():
    boot_ratio = ReplicateEstimator("bootstrap", "ratio")
    boot_ratio.estimate(
        y_boot,
        sample_wgt_boot,
        rep_wgt_boot,
        x=x_boot,
        conservative=True,
        exclude_nan=True,
    )
    boot_var = boot_ratio.variance
    boot_stderr = pow(boot_var.get("__none__"), 0.5)
    assert np.isclose(boot_stderr, 0.0005171, atol=1e-7)


def test_boot_ratio_d():
    boot_ratio_d = ReplicateEstimator("bootstrap", "ratio")
    boot_ratio_d.estimate(
        y_boot,
        sample_wgt_boot,
        rep_wgt_boot,
        x=x_boot,
        domain=domain_boot,
        conservative=True,
        exclude_nan=True,
    )
    boot_var_d = boot_ratio_d.variance
    boot_stderr_d1 = pow(boot_var_d.get(1), 0.5)
    boot_stderr_d2 = pow(boot_var_d.get(2), 0.5)
    boot_stderr_d3 = pow(boot_var_d.get(3), 0.5)
    boot_stderr_d4 = pow(boot_var_d.get(4), 0.5)
    boot_stderr_d5 = pow(boot_var_d.get(5), 0.5)
    assert np.isclose(boot_stderr_d1, 0.0015745, atol=1e-7)
    assert np.isclose(boot_stderr_d2, 0.0009610, atol=1e-7)
    assert np.isclose(boot_stderr_d3, 0.0009393, atol=1e-7)
    assert np.isclose(boot_stderr_d4, 0.0012156, atol=1e-7)
    assert np.isclose(boot_stderr_d5, 0.0021456, atol=1e-7)
