import pytest

import numpy as np

from samplics.utils.basic_functions import (
    averageby,
    kurtosis,
    plot_kurtosis,
    plot_skewness,
    skewness,
    sumby,
    transform,
    averageby,
    skewness,
    kurtosis,
    plot_skewness,
    plot_kurtosis,
)

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
    assert np.isclose(y_transformed, np.exp(np.log(1 + (y + constant) * llambda) / llambda)).all()


@pytest.mark.xfail
@pytest.mark.parametrize("y", [y1, y2])
def test_plot_skewness(y):
    plot_skewness(y, block=False)


@pytest.mark.xfail
@pytest.mark.parametrize("y", [y1, y2])
def test_plot_kurtosis(y):
    plot_kurtosis(y, block=False)


@pytest.mark.parametrize("y", [y1, y2])
def test_plot_skewness(y):
    plot_skewness(y, coef_min=-2, coef_max=3, block=False)


@pytest.mark.parametrize("y", [y1, y2])
def test_plot_kurtosis(y):
    plot_kurtosis(y, block=False)
