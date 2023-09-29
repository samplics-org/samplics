import numpy as np
import pandas as pd
import polars as pl
import pytest

from samplics.types.data import AuxVars


# Auxiliary variables


class TestAuxVars:
    # Create an instance of AuxVars with a numpy array as area and a pandas dataframe as x
    def test_create_instance_with_numpy_array_and_pandas_dataframe(self):
        area = np.array([1, 2, 2, 1, 1])
        x = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]})
        aux_vars = AuxVars(x, area)
        breakpoint()
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.x, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2]
        assert aux_vars.x == {
            1: {"record_id": [0, 3, 4], "col1": [1, 4, 5], "col2": [6, 9, 10]},
            2: {"record_id": [1, 2], "col1": [2, 3], "col2": [7, 8]},
        }
        # assert aux_vars.ssize == {1: 3, 2: 2}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )

    # Create an instance of AuxVars with a list as area and a numpy array as x
    def test_create_instance_with_list_and_numpy_array(self):
        area = [1, 2, 3, 4, 5]
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        aux_vars = AuxVars(area, x)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.x, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2, 3, 4, 5]
        assert aux_vars.x == {
            1: {"record_id": [0], "__x_0": [1], "__x_1": [2], "__x_2": [3]},
            2: {"record_id": [1], "__x_0": [4], "__x_1": [5], "__x_2": [6]},
            3: {"record_id": [2], "__x_0": [7], "__x_1": [8], "__x_2": [9]},
            4: {"record_id": [3], "__x_0": [10], "__x_1": [11], "__x_2": [12]},
            5: {"record_id": [4], "__x_0": [13], "__x_1": [14], "__x_2": [15]},
        }
        # assert aux_vars.ssize == {}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )

    # Create an instance of AuxVars with a tuple as area and a polars dataframe as x
    def test_create_instance_with_tuple_and_polars_dataframe(self):
        area = (1, 2, 3, 4, 5)
        x = pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]})
        aux_vars = AuxVars(area, x)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.x, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2, 3, 4, 5]
        assert aux_vars.x == {
            1: {"record_id": [0], "col1": [1], "col2": [6]},
            2: {"record_id": [1], "col1": [2], "col2": [7]},
            3: {"record_id": [2], "col1": [3], "col2": [8]},
            4: {"record_id": [3], "col1": [4], "col2": [9]},
            5: {"record_id": [4], "col1": [5], "col2": [10]},
        }
        # assert aux_vars.ssize == {}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )

    # Create an instance of AuxVars with an empty numpy array as area
    # and a pandas dataframe as x
    @pytest.mark.xfail(reason="Empty area array not allowed")
    def test_create_instance_with_empty_numpy_array_and_pandas_dataframe(self):
        area = np.array([])
        x = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]})
        aux_vars = AuxVars(area, x)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.x, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == []
        assert aux_vars.x == {
            1: {"record_id": [0], "col1": [1], "col2": [6]},
            2: {"record_id": [1], "col1": [2], "col2": [7]},
            3: {"record_id": [2], "col1": [3], "col2": [8]},
            4: {"record_id": [3], "col1": [4], "col2": [9]},
            5: {"record_id": [4], "col1": [5], "col2": [10]},
        }
        # assert aux_vars.ssize == {}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )

    # Create an instance of AuxVars with a numpy array as area
    # and an empty polars dataframe as x
    @pytest.mark.xfail(reason="Empty auxiliary data not allowed")
    def test_create_instance_with_numpy_array_and_empty_polars_dataframe(self):
        area = np.array([1, 2, 3, 4, 5])
        x = pl.DataFrame()
        aux_vars = AuxVars(area, x)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.x, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2, 3, 4, 5]
        assert aux_vars.x == {}
        # assert aux_vars.ssize == {}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )

    # Create an instance of AuxVars with a numpy array as area and a numpy array as x
    def test_create_instance_with_numpy_array_and_numpy_array(self):
        area = np.array([1, 2, 3, 4, 5])
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        aux_vars = AuxVars(area, x)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.x, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2, 3, 4, 5]
        assert aux_vars.x == {
            1: {"record_id": [0], "__x_0": [1], "__x_1": [2], "__x_2": [3]},
            2: {"record_id": [1], "__x_0": [4], "__x_1": [5], "__x_2": [6]},
            3: {"record_id": [2], "__x_0": [7], "__x_1": [8], "__x_2": [9]},
            4: {"record_id": [3], "__x_0": [10], "__x_1": [11], "__x_2": [12]},
            5: {"record_id": [4], "__x_0": [13], "__x_1": [14], "__x_2": [15]},
        }
        # assert aux_vars.ssize == {}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )
