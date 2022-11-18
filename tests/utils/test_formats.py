import pytest

import numpy as np
import pandas as pd

from samplics.utils.formats import (
    _array_to_dict,
    convert_numbers_to_dicts,
    _dataframe_to_array,
    _dict_to_dataframe,
    _numpy_array,
    sample_size_dict,
    _sample_units,
)


df = pd.DataFrame({"one": [1, 2, 2, 3, 0], "two": [4, 9, 5, 6, 6]})
ds = pd.Series([1, 2, 3, 4, 5])
da = np.array([1, 2, 3, 4, 5])
da2 = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("input_data", [df, ds, da, da2])
def test_numpy_array(input_data):
    arr = _numpy_array(input_data)
    assert isinstance(arr, np.ndarray) == True


@pytest.mark.parametrize("arr", [da])
@pytest.mark.parametrize("domain", [da, None])
def test_array_to_dict1(arr, domain):
    dictionary = _array_to_dict(arr, domain)
    assert isinstance(dictionary, dict) == True


@pytest.mark.parametrize("input_df", [df, ds])
def test_dataframe_to_array(input_df):
    arr = _dataframe_to_array(input_df)
    assert isinstance(arr, np.ndarray) == True


stratum = ["one", "two", 3, 4, 5]
samp_size = {"one": 11, "two": 22, 3: 33, 4: 44, 5: 55}


def test_sample_size_dict1(sample_size=5, stratification=False, stratum=stratum):
    samp_dict1 = sample_size_dict(sample_size, stratification, stratum)
    assert isinstance(samp_dict1, int) == True
    assert samp_dict1 == 5


def test_sample_size_dict1(sample_size=5, stratification=False, stratum=stratum):
    samp_dict2 = sample_size_dict(sample_size, stratification, stratum)
    assert isinstance(samp_dict2, int) == True
    assert samp_dict2 == 5


def test_sample_size_dict_str1(sample_size=5, stratification=True, stratum=stratum):
    samp_dict_str1 = sample_size_dict(sample_size, stratification, stratum)
    assert isinstance(samp_dict_str1, dict) == True
    assert samp_dict_str1["one"] == 5
    assert samp_dict_str1["two"] == 5
    assert samp_dict_str1[3] == 5
    assert samp_dict_str1[4] == 5
    assert samp_dict_str1[5] == 5


def test_sample_size_dict_str2(sample_size=samp_size, stratification=True, stratum=stratum):
    samp_dict_str2 = sample_size_dict(sample_size, stratification, stratum)
    assert isinstance(samp_dict_str2, dict) == True
    assert samp_dict_str2["one"] == 11
    assert samp_dict_str2["two"] == 22
    assert samp_dict_str2[3] == 33
    assert samp_dict_str2[4] == 44
    assert samp_dict_str2[5] == 55


def test_sample_units1(all_units=da, unique=True):
    all_units = _sample_units(all_units, unique)
    assert all_units.size == 5


def test_sample_units2(all_units=da2, unique=True):
    all_units = _sample_units(all_units, unique)
    assert all_units.size == 5


def test_sample_units3(all_units=[1, 2, 2, 3, 3, 3], unique=False):
    all_units = _sample_units(all_units, unique)
    assert all_units.size == 6


def test_dict_to_dataframe():
    df = _dict_to_dataframe(["parameter", "domain", "size"], samp_size)
    assert isinstance(df, pd.DataFrame) == True
    assert (df.columns == ["parameter", "domain", "size"]).all()


dict1 = {"one": 1, "two": 2, "three": 3}
dict2 = {"one": 11, "two": 22, "three": 33}
dict3 = {"one": 11, "two": 22}
dict4 = {"one": 11, "three": 22}


def test_convert_numbers_to_dicts1():
    dicts = convert_numbers_to_dicts(3, 100, 1000, 10000)
    assert dicts[0]["_stratum_1"] == 100
    assert dicts[0]["_stratum_2"] == 100
    assert dicts[0]["_stratum_3"] == 100
    assert dicts[1]["_stratum_1"] == 1000
    assert dicts[1]["_stratum_2"] == 1000
    assert dicts[1]["_stratum_3"] == 1000
    assert dicts[2]["_stratum_1"] == 10000
    assert dicts[2]["_stratum_2"] == 10000
    assert dicts[2]["_stratum_3"] == 10000


def test_convert_numbers_to_dicts2():
    dicts = convert_numbers_to_dicts(3, 100, dict1, dict2)
    assert dicts[0]["one"] == 100
    assert dicts[0]["two"] == 100
    assert dicts[0]["three"] == 100
    assert dicts[1]["one"] == 1
    assert dicts[1]["two"] == 2
    assert dicts[1]["three"] == 3
    assert dicts[2]["one"] == 11
    assert dicts[2]["two"] == 22
    assert dicts[2]["three"] == 33


def test_convert_numbers_to_dicts3():
    dicts = convert_numbers_to_dicts(3, dict1, dict2)
    assert dicts[0]["one"] == 1
    assert dicts[0]["two"] == 2
    assert dicts[0]["three"] == 3
    assert dicts[1]["one"] == 11
    assert dicts[1]["two"] == 22
    assert dicts[1]["three"] == 33


@pytest.mark.xfail(reason="dictionnaries have different keys")
def test_convert_numbers_to_dicts4():
    convert_numbers_to_dicts(3, dict1, dict3)


@pytest.mark.xfail(reason="dictionnaries have different keys")
def test_convert_numbers_to_dicts5():
    convert_numbers_to_dicts(3, dict3, dict4)
