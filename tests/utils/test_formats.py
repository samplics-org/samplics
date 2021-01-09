import pytest

import numpy as np
import pandas as pd

from samplics.utils.formats import *


df = pd.DataFrame({"one": [1, 2, 2, 3, 0], "two": [4, 9, 5, 6, 6]})
ds = pd.Series([1, 2, 3, 4, 5])
da = np.array([1, 2, 3, 4, 5])
da2 = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("input_data", [df, ds, da, da2])
def test_numpy_array(input_data):
    arr = numpy_array(input_data)
    assert isinstance(arr, np.ndarray) == True


@pytest.mark.parametrize("arr", [da])
@pytest.mark.parametrize("domain", [da, None])
def test_array_to_dict1(arr, domain):
    dictionary = array_to_dict(arr, domain)
    assert isinstance(dictionary, dict) == True


@pytest.mark.parametrize("input_df", [df, ds])
def test_dataframe_to_array(input_df):
    arr = dataframe_to_array(input_df)
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
    all_units = sample_units(all_units, unique)
    assert all_units.size == 5


def test_sample_units2(all_units=da2, unique=True):
    all_units = sample_units(all_units, unique)
    assert all_units.size == 5


def test_sample_units3(all_units=[1, 2, 2, 3, 3, 3], unique=False):
    all_units = sample_units(all_units, unique)
    assert all_units.size == 6


def test_dict_to_dataframe():
    df = dict_to_dataframe(["domain", "size"], samp_size)
    assert isinstance(df, pd.DataFrame) == True
    assert (df.columns == ["domain", "size"]).all()
