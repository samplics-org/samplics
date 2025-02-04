import numpy as np
import pandas as pd
import pytest

from samplics.utils.basic_functions import (
    averageby,
    get_single_psu_strata,
    kurtosis,
    plot_kurtosis,
    plot_skewness,
    set_variables_names,
    skewness,
    sumby,
    transform,
)


def test_set_variables_names1():
    vars = np.array([1, 2, 3])

    name1 = set_variables_names(vars=vars, varnames=["one"], prefix="var")
    assert name1 == ["one"]

    name2 = set_variables_names(vars=vars, varnames=None, prefix="var")
    assert name2 == ["var_1"]


def test_set_variables_names2():
    vars = pd.DataFrame(data={"one": [1, 2, 3], "two": [4, 5, 6]})

    name1 = set_variables_names(vars=vars, varnames=None, prefix="var")
    assert name1 == ["one", "two"]

    name2 = set_variables_names(vars=vars, varnames=["1", "2"], prefix="var")
    assert name2 == ["1", "2"]


def test_set_variables_names_series_with_no_name():
    vars = pd.Series(data={"one": [1, 2, 3]})

    name1 = set_variables_names(vars=vars, varnames=None, prefix="var")
    assert name1 == ["var_1"]

    name2 = set_variables_names(vars=vars, varnames=["1"], prefix="var")
    assert name2 == ["1"]


def test_set_variables_names_series_with_name():
    vars = pd.Series(data={"one": [1, 2, 3]}, name="one")

    name1 = set_variables_names(vars=vars, varnames=None, prefix="None")
    assert name1 == ["one"]

    name2 = set_variables_names(vars=vars, varnames=["1"], prefix="None")
    assert name2 == ["1"]


def test_set_variables_names_numpy():
    vars1 = np.array([[1, 2], [3, 4]])
    name1 = set_variables_names(vars=vars1, varnames=None, prefix="var")
    assert name1 == ["var_1", "var_2"]

    vars2 = np.array([[1, 2, 3], [4, 5, 6]])
    name2 = set_variables_names(vars=vars2, varnames=None, prefix="var")
    assert name2 == ["var_1", "var_2", "var_3"]

    vars3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    name3 = set_variables_names(vars=vars3, varnames=None, prefix="var")
    assert name3 == ["var_1", "var_2", "var_3", "var_4"]


group1 = np.array([1, 2, 1, 3, 2, 1, 2, 2, 1, 3, 2])
y1 = np.array([5, 3, 1, 2, 2, 3, 1, 3, 6, 21, 7])

group2 = np.array([1, 2, 1, 3, 2, 1, 2, 2, 1, 3, 2])
y2 = np.array([1, -13, 1, 2, -12, 0, 1, 3, 16, -21, 77])


def test_sumby1():
    assert (sumby(group1, y1) == [15, 16, 23]).all()


def test_sumby2():
    assert (sumby(group2, y2) == [18, 56, -19]).all()


def test_averageby1():
    assert (averageby(group1, y1) == [3.75, 3.2, 11.5]).all()


def test_averageby2():
    assert (averageby(group2, y2) == [4.5, 11.2, -9.5]).all()


def test_skewness1():
    assert np.isclose(skewness(y1), 2.278926)
    assert np.isclose(skewness(y1, type=2), 2.655731)
    assert np.isclose(skewness(y1, type=3), 1.975337)


def test_skewness2():
    assert np.isclose(skewness(y2), 2.101308)
    assert np.isclose(skewness(y2, type=2), 2.448745)
    assert np.isclose(skewness(y2, type=3), 1.821391)


def test_kurtosis1():
    assert np.isclose(kurtosis(y1), 4.142948)
    assert np.isclose(kurtosis(y1, type=2), 7.738246)
    assert np.isclose(kurtosis(y1, type=3), 2.903263)


def test_kurtosis2():
    assert np.isclose(kurtosis(y2), 3.787848)
    assert np.isclose(kurtosis(y2, type=2), 7.146413)
    assert np.isclose(kurtosis(y2, type=3), 2.609792)


@pytest.mark.parametrize("y", [y1, y2])
def test_transform_log(y):
    constant = -(min(y) - 10)
    y_transformed = transform(y, 0, constant, inverse=False)
    assert np.isclose(y_transformed, np.log(y + constant)).all()


@pytest.mark.parametrize("y", [y1, y2])
def test_transform_log_inverse(y):
    constant = -(min(y) - 10)
    y_transformed = transform(y, 0, constant, inverse=True)
    assert np.isclose(y_transformed, np.exp(y) - constant).all()


@pytest.mark.parametrize("y", [y1, y2])
def test_transform_exp(y):
    constant = -min(y) + 10
    llambda = 2
    y_transformed = transform(y + constant, llambda, constant, inverse=False)
    assert np.isclose(y_transformed, np.power(y + constant, llambda) / llambda).all()


@pytest.mark.parametrize("y", [y1, y2])
def test_transform_exp_inverse(y):
    constant = -min(y) + 10
    llambda = 2
    y_transformed = transform(y + constant, llambda, constant, inverse=True)
    assert np.isclose(
        y_transformed, np.exp(np.log(1 + (y + constant) * llambda) / llambda)
    ).all()


# # @pytest.mark.xfail(strict=True)
# @pytest.mark.parametrize("y", [y1, y2])
# def test_plot_skewness1(y):
#     plot_skewness(y, block=False)


# # @pytest.mark.xfail(strict=True)
# @pytest.mark.parametrize("y", [y1, y2])
# def test_plot_kurtosis1(y):
#     plot_kurtosis(y, block=False)


@pytest.mark.parametrize("y", [y1, y2])
def test_plot_skewness2(y):
    plot_skewness(y, coef_min=-2, coef_max=3, block=False)


@pytest.mark.parametrize("y", [y1, y2])
def test_plot_kurtosis2(y):
    plot_kurtosis(y, block=False)


strat = np.array([1, 2, 2, 1, 2, 3, 3, 41, 1, 2, 1, 2, 3, 55])
psu = np.array([1, 2, 1, 1, 2, 3, 1, 4, 2, 1, np.nan, 2, 3, np.nan])


def test_single_psu():
    single_psu_strata = get_single_psu_strata(stratum=strat, psu=psu)
    assert 41 in single_psu_strata
    assert 55 in single_psu_strata
