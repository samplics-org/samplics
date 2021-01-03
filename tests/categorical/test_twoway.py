import pytest
import numpy as np
import pandas as pd
from samplics.categorical.tabulation import TwoWay

from samplics.estimation import TaylorEstimator
from samplics.categorical import OneWay


birthcat = pd.read_csv("./tests/categorical/birthcat.csv")

region = birthcat["region"].to_numpy().astype(int)
age_cat = birthcat["agecat"].to_numpy()
birth_cat = birthcat["birthcat"].to_numpy()
pop = birthcat["pop"]


@pytest.mark.xfail(strict=True, reason="Parameter not valid")
@pytest.mark.parametrize("param", ["total", "mean", "ratio", "other"])
def test_not_valid_parameter(param):
    tbl = TwoWay(param)


# @pytest.mark.xfail(strict=True, reason="2way tables needs two variables")
# def test_twoway_count_one_var_count():
#     tbl = TwoWay("count")
#     tbl.tabulate(region, remove_nan=True)


# tbl_count = TwoWay("count")
# tbl_count.tabulate([region, birth_cat], remove_nan=True)

tbl_prop = TwoWay("proportion")
tbl_prop.tabulate([age_cat, birth_cat], varnames=["age_cat", "birth_cat"], remove_nan=True)

breakpoint()

tbl_count = TwoWay("count")
tbl_count.tabulate([age_cat, birth_cat], varnames=["age_cat", "birth_cat"], remove_nan=True)


def test_oneway_count_one_var_count():
    assert tbl_count.table["birthcat"][1] == 240
    assert tbl_count.table["birthcat"][2] == 450
    assert tbl_count.table["birthcat"][3] == 233