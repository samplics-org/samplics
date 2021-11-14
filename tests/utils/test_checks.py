import pytest

import numpy as np
import pandas as pd


from samplics.utils.checks import assert_in_range


def test_in_range_ints_successes():
    assert assert_in_range(low=10, high=39, x=35)
    assert assert_in_range(low=-39, high=-10, x=-35)
    assert assert_in_range(low=0, high=1, x=0.35)
    assert assert_in_range(low=-10, high=39, x=0, y=1, z=39)


def test_in_range_for_ints_fails():
    assert not assert_in_range(low=10, high=39, x=5)
    assert not assert_in_range(low=-39, high=-10, x=-135)
    assert not assert_in_range(low=0, high=1, x=1.35)
    assert not assert_in_range(low=-10, high=39, x=0, y=100, z=39)


def test_fin_range_for_floats_successes():
    assert assert_in_range(low=10.1, high=39.0, x=35.22)
    assert assert_in_range(low=-39.2, high=-10.1, x=-35.35)
    assert assert_in_range(low=0.0, high=1.0, x=0.35)
    assert assert_in_range(low=-10.0, high=39.0, x=0.0, y=1.9, z=39.0)


def test_in_range_for_floats_fails():
    assert not assert_in_range(low=10.0, high=39.0, x=5.5)
    assert not assert_in_range(low=-39.3, high=-10.1, x=-135.23)
    assert not assert_in_range(low=0.0, high=1.00, x=1.35)
    assert not assert_in_range(low=-10.0, high=39.33, x=0.7, y=100.01, z=39.3)


def test_in_range_np_pd_successes():
    assert assert_in_range(low=11, high=23, x=np.array([20, 17, 11, 20, 23]))
    assert assert_in_range(low=11, high=23, x=pd.Series([20, 17, 11, 20, 23]))
    assert assert_in_range(low=-11, high=23, x=np.array([-2, 17, -11, 20, 23]))
    assert assert_in_range(low=-11, high=23, x=pd.Series([-10, 17, -11, 20, 23]))


def test_in_range_for_np_pd_fails():
    assert not assert_in_range(low=11, high=23, x=np.array([20, 17, 111, 20, 23]))
    assert not assert_in_range(low=11, high=23, x=pd.Series([20, -107, 11, 20, 23]))
    assert not assert_in_range(low=-101, high=0, x=np.array([-2, -17, -11, -20, 0.0023]))
    assert not assert_in_range(low=-11, high=23, x=pd.Series([-10, 17, -11.0001, 20, 23]))


def test_np_list_in_range_successes():
    assert assert_in_range(low=11, high=23, x=[20, 17, 11, 20, 23])
    assert assert_in_range(low=11, high=23, x=[20, 17, 11, 20, 23])
    assert assert_in_range(low=-11, high=23, x=[-2, 17, -11, 20, 23])
    assert assert_in_range(low=-11, high=23, x=[-10, 17, -11, 20, 23])


def test_list_in_range_fails():
    assert not assert_in_range(low=11, high=23, x=np.array([20, 17, 111, 20, 23]))
    assert not assert_in_range(low=11, high=23, x=pd.Series([20, -107, 11, 20, 23]))
    assert not assert_in_range(low=-101, high=0, x=np.array([-2, -17, -11, -20, 0.0023]))
    assert not assert_in_range(low=-11, high=23, x=pd.Series([-10, 17, -11.0001, 20, 23]))
