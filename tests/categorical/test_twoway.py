import pytest
import numpy as np
import pandas as pd

from samplics.estimation import TaylorEstimator
from samplics.categorical import OneWay, CrossTabulation


birthcat = pd.read_csv("./tests/categorical/birthcat.csv")

# birthcat.loc[(birthcat["birthcat"] == 2) & (birthcat["region"]==3), "birthcat"] = 3

region = birthcat["region"].to_numpy().astype(int)
age_cat = birthcat["agecat"].to_numpy()
birth_cat = birthcat["birthcat"].to_numpy()
pop = birthcat["pop"]


@pytest.mark.xfail(strict=True, reason="Parameter not valid")
@pytest.mark.parametrize("param", ["total", "mean", "ratio", "other"])
def test_not_valid_parameter(param):
    tbl = CrossTabulation(param)


# @pytest.mark.xfail(strict=True, reason="2way tables needs two variables")
# def test_twoway_count_one_var_count():
#     tbl = CrossTabulation("count")
#     tbl.tabulate(region, remove_nan=True)


@pytest.mark.xfail(strict=True, reason="For now, the method will fail if there are missing values")
def test_for_missing_values_in_the_design_matrix():
    tbl_prop = CrossTabulation("proportion")
    tbl_prop.tabulate([region, birth_cat], varnames=["region", "birth_cat"], remove_nan=False)


# tbl_count = CrossTabulation("count")
# tbl_count.tabulate([region, birth_cat], remove_nan=True)

tbl_prop = CrossTabulation("proportion")
tbl_prop.tabulate([region, birth_cat], varnames=["region", "birth_cat"], remove_nan=True)

breakpoint()

tbl_count = CrossTabulation("count")
tbl_count.tabulate([age_cat, birth_cat], varnames=["age_cat", "birth_cat"], remove_nan=True)


def test_oneway_count_one_var_count():
    assert tbl_count.table["birthcat"][1] == 240
    assert tbl_count.table["birthcat"][2] == 450
    assert tbl_count.table["birthcat"][3] == 233