import pytest
import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator
from samplics.categorical import OneWay

birthcat = pd.read_csv("./tests/categorical/birthcat.csv")

region = birthcat["region"]
age_cat = birthcat["agecat"]
birth_cat = birthcat["birthcat"]
pop = birthcat["pop"]


tbl_count = OneWay("count")
# print(tbl_count)


@pytest.mark.xfail(strict=True, reason="Parameter not valid")
@pytest.mark.parametrize("param", ["total", "mean", "ratio", "other"])
def test_not_valid_parameter(param):
    tbl = OneWay(param)


# def test_one():
#     tbl_count.tabulate(birth_cat)
#     assert(tbl_count.table, nan)


# def test_one():
#     tbl_count.tabulate(birth_cat, remove_nan=True)

def test_two():
    tbl_count.tabulate([birth_cat, region], remove_nan=True)