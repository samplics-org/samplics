import pytest

import numpy as np
import pandas as pd

from samplics.utils import checks, formats

from samplics.sampling import SampleSize

## Wald's method

size_nat_wald = SampleSize()


def test_size_nat_wald_basics():
    assert size_nat_wald.parameter == "proportion"
    assert size_nat_wald.method == "wald"
    assert size_nat_wald.stratification == False


def test_size_nat_wald_size():
    size_nat_wald.allocate(0.80, 0.10)
    assert size_nat_wald.samp_size["__none__"] == 62


def test_size_nat_wald_size_with_deff():
    size_nat_wald.allocate(0.80, 0.10, deff=1.5)
    assert size_nat_wald.samp_size["__none__"] == 93


## Wald's method - stratified
size_str_wald = SampleSize(parameter="Proportion", method="Wald", stratification=True)

target = {"stratum1": 0.95, "stratum2": 0.70, "stratum3": 0.30}
precision = {"stratum1": 0.30, "stratum2": 0.10, "stratum3": 0.15}
deff = {"stratum1": 1, "stratum2": 1.5, "stratum3": 2.5}


def test_size_str_wald_basics():
    assert size_str_wald.parameter == "proportion"
    assert size_str_wald.method == "wald"
    assert size_str_wald.stratification == True


def test_size_str_wald_size1():
    size_str_wald.allocate(target, 0.10)
    assert size_str_wald.samp_size["stratum1"] == 19
    assert size_str_wald.samp_size["stratum2"] == 81
    assert size_str_wald.samp_size["stratum3"] == 81


def test_size_str_wald_size2():
    size_str_wald.allocate(0.8, precision)
    assert size_str_wald.samp_size["stratum1"] == 7
    assert size_str_wald.samp_size["stratum2"] == 62
    assert size_str_wald.samp_size["stratum3"] == 28


def test_size_str_wald_size3():
    size_str_wald.allocate(0.8, 0.10, deff)
    assert size_str_wald.samp_size["stratum1"] == 62
    assert size_str_wald.samp_size["stratum2"] == 93
    assert size_str_wald.samp_size["stratum3"] == 154


def test_size_str_wald_size4():
    size_str_wald.allocate(target, precision, deff)
    assert size_str_wald.samp_size["stratum1"] == 3
    assert size_str_wald.samp_size["stratum2"] == 122
    assert size_str_wald.samp_size["stratum3"] == 90
