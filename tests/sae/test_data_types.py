import numpy as np
import pandas as pd
import polars as pl
import pytest

from samplics.sae.data_types import AuxVars, DirectEst, FitMethod


# FitMethod


class TestFitMethod1:
    def test_fh_enum(self):
        fh_method = FitMethod.fh
        assert fh_method.name == "fh"
        assert fh_method.value == "FH"

    def test_ml_enum(self):
        ml_method = FitMethod.ml
        assert ml_method.name == "ml"
        assert ml_method.value == "ML"

    def test_reml_enum(self):
        reml_method = FitMethod.reml
        assert reml_method.name == "reml"
        assert reml_method.value == "REML"


# Direct Estimator


class TestDirectEst1:
    def test_direct_est0(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est=1,  # {"one": 1, "two": 2},
            stderr=2,
            ssize=3,
        )
        assert est.areas == ["one", "two", "three", "four", "five"]
        assert est.est == {"one": 1, "two": 1, "three": 1, "four": 1, "five": 1}
        assert est.stderr == {"one": 2, "two": 2, "three": 2, "four": 2, "five": 2}
        assert est.ssize == {"one": 3, "two": 3, "three": 3, "four": 3, "five": 3}

    def test_direct_est1(self):
        est = DirectEst(
            area=("one", "two", "three", "four", "five"),
            est=[1, 2, 3, 4, 5],  # {"one": 1, "two": 2},
            stderr=[0.1, 0.2, 0.3, 0.4, 0.5],
            ssize={"one": 10, "two": 20, "three": 30, "four": 40, "five": 50},
        )
        assert est.areas == ["one", "two", "three", "four", "five"]
        assert est.est == {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        assert est.stderr == {"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5}
        assert est.ssize == {"one": 10, "two": 20, "three": 30, "four": 40, "five": 50}

    def test_direct_est2(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est=[1, 2, 3, 4, 5],  # {"one": 1, "two": 2},
            stderr=[0.1, 0.2, 0.3, 0.4, 0.5],
            ssize={"one": 10, "two": 20, "three": 30, "four": 40, "five": 50},
        )
        assert est.areas == ["one", "two", "three", "four", "five"]
        assert est.est == {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        assert est.stderr == {"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5}
        assert est.ssize == {"one": 10, "two": 20, "three": 30, "four": 40, "five": 50}

    def test_direct_est3(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est={"one": 1, "two": 2, "three": 3, "four": 4, "five": 5},
            stderr={"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5},
            ssize=35,
        )
        assert est.areas == ["one", "two", "three", "four", "five"]
        assert est.est == {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        assert est.stderr == {"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5}
        assert est.ssize == {"one": 35, "two": 35, "three": 35, "four": 35, "five": 35}

    def test_directest_cv(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est={"one": 1, "two": 2, "three": 3, "four": 4, "five": 5},
            stderr={"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5},
            ssize=35,
        )
        assert est.cv == {
            "one": 0.1,
            "two": 0.2 / 2,
            "three": 0.3 / 3,
            "four": 0.4 / 4,
            "five": 0.5 / 5,
        }

    def test_directest_numpy(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est={"one": 1, "two": 2, "three": 3, "four": 4, "five": 5},
            stderr={"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5},
            ssize=35,
        )

        est_np = est.to_numpy()
        assert est.areas == list(est_np[:, 0])
        assert est.est == dict(zip(est.areas, est_np[:, 1]))
        assert est.stderr == dict(zip(est.areas, est_np[:, 2]))
        assert est.ssize == dict(zip(est.areas, est_np[:, 3]))

    def test_directest_polars(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est=1,  # {"one": 1, "two": 2},
            stderr=2,
            ssize=3,
        )

        est_pl = est.to_polars()
        assert (est.areas == est_pl.select("areas").to_numpy().flatten()).all()
        assert est.est == dict(zip(est.areas, est_pl.select("est").to_numpy().flatten()))
        assert est.stderr == dict(zip(est.areas, est_pl.select("stderr").to_numpy().flatten()))
        assert est.ssize == dict(zip(est.areas, est_pl.select("ssize").to_numpy().flatten()))

    def test_directest_pandas(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est=[1, 2, 3, 4, 5],  # {"one": 1, "two": 2},
            stderr=[0.1, 0.2, 0.3, 0.4, 0.5],
            ssize={"one": 10, "two": 20, "three": 30, "four": 40, "five": 50},
        )

        est_pd = est.to_pandas()
        assert (est.areas == est_pd["areas"].values).all()
        assert est.est == dict(zip(est.areas, est_pd["est"].values))
        assert est.stderr == dict(zip(est.areas, est_pd["stderr"].values))
        assert est.ssize == dict(zip(est.areas, est_pd["ssize"].values))


class TestDirectEst2:
    # Tests that an instance of DirectEst can be created with valid inputs.
    def test_create_instance_with_valid_inputs(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(area, est, stderr, ssize, psize)

        assert direct_est.areas == [1, 2, 3]
        assert direct_est.est == {1: 10, 2: 20, 3: 30}
        assert direct_est.stderr == {1: 1, 2: 2, 3: 3}
        assert direct_est.ssize == {1: 100, 2: 200, 3: 300}
        assert direct_est.psize == {1: 1000, 2: 2000, 3: 3000}

    # Tests that the cv property of an instance of DirectEst can be accessed.
    def test_access_cv_property(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(area, est, stderr, ssize, psize)

        cv = direct_est.cv

        assert cv == {1: 0.1, 2: 0.1, 3: 0.1}

    # Tests that an instance of DirectEst can be converted to a numpy array.
    def test_convert_to_numpy_array(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(area, est, stderr, ssize, psize)

        numpy_array = direct_est.to_numpy()

        assert numpy_array.shape == (3, 4)

    # Tests that an instance of DirectEst can be converted to a polars dataframe.
    def test_convert_to_polars_dataframe(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(area, est, stderr, ssize, psize)

        polars_df = direct_est.to_polars()

        assert polars_df.shape == (3, 4)

    # Tests that an instance of DirectEst can be converted to a pandas dataframe.
    def test_convert_to_pandas_dataframe(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(area, est, stderr, ssize, psize)

        pandas_df = direct_est.to_pandas()

        assert pandas_df.shape == (3, 4)

    # Tests that an instance of DirectEst can be created with a single area and
    # a None value as the psize parameter.
    def test_create_instance_with_single_area_and_none_psize(self):
        area = [1]
        est = {1: 10}
        stderr = {1: 1}
        ssize = {1: 100}
        psize = None

        direct_est = DirectEst(area, est, stderr, ssize, psize)

        assert direct_est.areas == [1]
        assert direct_est.est == {1: 10}
        assert direct_est.stderr == {1: 1}
        assert direct_est.ssize == {1: 100}
        assert direct_est.psize is None


# Auxiliary variables


class TestAuxVars:
    # Create an instance of AuxVars with a numpy array as area and a pandas dataframe as auxdata
    def test_create_instance_with_numpy_array_and_pandas_dataframe(self):
        area = np.array([1, 2, 2, 1, 1])
        auxdata = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]})
        aux_vars = AuxVars(area, auxdata)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.auxdata, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2]
        assert aux_vars.auxdata == {
            1: {"record_id": [0, 3, 4], "col1": [1, 4, 5], "col2": [6, 9, 10]},
            2: {"record_id": [1, 2], "col1": [2, 3], "col2": [7, 8]},
        }
        # assert aux_vars.ssize == {1: 3, 2: 2}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )

    # Create an instance of AuxVars with a list as area and a numpy array as auxdata
    def test_create_instance_with_list_and_numpy_array(self):
        area = [1, 2, 3, 4, 5]
        auxdata = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        aux_vars = AuxVars(area, auxdata)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.auxdata, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2, 3, 4, 5]
        assert aux_vars.auxdata == {
            1: {"record_id": [0], "__aux_0": [1], "__aux_1": [2], "__aux_2": [3]},
            2: {"record_id": [1], "__aux_0": [4], "__aux_1": [5], "__aux_2": [6]},
            3: {"record_id": [2], "__aux_0": [7], "__aux_1": [8], "__aux_2": [9]},
            4: {"record_id": [3], "__aux_0": [10], "__aux_1": [11], "__aux_2": [12]},
            5: {"record_id": [4], "__aux_0": [13], "__aux_1": [14], "__aux_2": [15]},
        }
        # assert aux_vars.ssize == {}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )

    # Create an instance of AuxVars with a tuple as area and a polars dataframe as auxdata
    def test_create_instance_with_tuple_and_polars_dataframe(self):
        area = (1, 2, 3, 4, 5)
        auxdata = pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]})
        aux_vars = AuxVars(area, auxdata)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.auxdata, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2, 3, 4, 5]
        assert aux_vars.auxdata == {
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
    # and a pandas dataframe as auxdata
    @pytest.mark.xfail(reason="Empty area array not allowed")
    def test_create_instance_with_empty_numpy_array_and_pandas_dataframe(self):
        area = np.array([])
        auxdata = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]})
        aux_vars = AuxVars(area, auxdata)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.auxdata, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == []
        assert aux_vars.auxdata == {
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
    # and an empty polars dataframe as auxdata
    @pytest.mark.xfail(reason="Empty auxiliary data not allowed")
    def test_create_instance_with_numpy_array_and_empty_polars_dataframe(self):
        area = np.array([1, 2, 3, 4, 5])
        auxdata = pl.DataFrame()
        aux_vars = AuxVars(area, auxdata)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.auxdata, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2, 3, 4, 5]
        assert aux_vars.auxdata == {}
        # assert aux_vars.ssize == {}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )

    # Create an instance of AuxVars with a numpy array as area and a numpy array as auxdata
    def test_create_instance_with_numpy_array_and_numpy_array(self):
        area = np.array([1, 2, 3, 4, 5])
        auxdata = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        aux_vars = AuxVars(area, auxdata)
        assert isinstance(aux_vars, AuxVars)
        assert isinstance(aux_vars.areas, list)
        assert isinstance(aux_vars.auxdata, dict)
        # assert isinstance(aux_vars.ssize, dict)
        assert isinstance(aux_vars.uid, int)
        assert isinstance(aux_vars.uid, int)
        assert aux_vars.areas == [1, 2, 3, 4, 5]
        assert aux_vars.auxdata == {
            1: {"record_id": [0], "__aux_0": [1], "__aux_1": [2], "__aux_2": [3]},
            2: {"record_id": [1], "__aux_0": [4], "__aux_1": [5], "__aux_2": [6]},
            3: {"record_id": [2], "__aux_0": [7], "__aux_1": [8], "__aux_2": [9]},
            4: {"record_id": [3], "__aux_0": [10], "__aux_1": [11], "__aux_2": [12]},
            5: {"record_id": [4], "__aux_0": [13], "__aux_1": [14], "__aux_2": [15]},
        }
        # assert aux_vars.ssize == {}
        assert aux_vars.uid >= 0
        # assert aux_vars.uid <= int(
        #     dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d%H%M%S")
        #     + str(int(1e16 * rand.random()))
        # )
