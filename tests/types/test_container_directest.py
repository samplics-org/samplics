from samplics.types.containers import DirectEst

# Direct Estimator


class TestDirectEst1:
    def test_direct_est0(self):
        est = DirectEst(
            domain=["one", "two", "three", "four", "five"],
            est=1,  # {"one": 1, "two": 2},
            stderr=2,
            ssize=3,
        )
        assert est.domains == ["one", "two", "three", "four", "five"]
        assert est.est == {"one": 1, "two": 1, "three": 1, "four": 1, "five": 1}
        assert est.stderr == {"one": 2, "two": 2, "three": 2, "four": 2, "five": 2}
        assert est.ssize == {"one": 3, "two": 3, "three": 3, "four": 3, "five": 3}

    def test_direct_est1(self):
        est = DirectEst(
            domain=("one", "two", "three", "four", "five"),
            est=[1, 2, 3, 4, 5],  # {"one": 1, "two": 2},
            stderr=[0.1, 0.2, 0.3, 0.4, 0.5],
            ssize={"one": 10, "two": 20, "three": 30, "four": 40, "five": 50},
        )
        assert est.domains == ["one", "two", "three", "four", "five"]
        assert est.est == {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        assert est.stderr == {
            "one": 0.1,
            "two": 0.2,
            "three": 0.3,
            "four": 0.4,
            "five": 0.5,
        }
        assert est.ssize == {"one": 10, "two": 20, "three": 30, "four": 40, "five": 50}

    def test_direct_est2(self):
        est = DirectEst(
            domain=["one", "two", "three", "four", "five"],
            est=[1, 2, 3, 4, 5],  # {"one": 1, "two": 2},
            stderr=[0.1, 0.2, 0.3, 0.4, 0.5],
            ssize={"one": 10, "two": 20, "three": 30, "four": 40, "five": 50},
        )
        assert est.domains == ["one", "two", "three", "four", "five"]
        assert est.est == {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        assert est.stderr == {
            "one": 0.1,
            "two": 0.2,
            "three": 0.3,
            "four": 0.4,
            "five": 0.5,
        }
        assert est.ssize == {"one": 10, "two": 20, "three": 30, "four": 40, "five": 50}

    def test_direct_est3(self):
        est = DirectEst(
            domain=["one", "two", "three", "four", "five"],
            est={"one": 1, "two": 2, "three": 3, "four": 4, "five": 5},
            stderr={"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5},
            ssize=35,
        )
        assert est.domains == ["one", "two", "three", "four", "five"]
        assert est.est == {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        assert est.stderr == {
            "one": 0.1,
            "two": 0.2,
            "three": 0.3,
            "four": 0.4,
            "five": 0.5,
        }
        assert est.ssize == {"one": 35, "two": 35, "three": 35, "four": 35, "five": 35}

    def test_directest_cv(self):
        est = DirectEst(
            domain=["one", "two", "three", "four", "five"],
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
            domain=["one", "two", "three", "four", "five"],
            est={"one": 1, "two": 2, "three": 3, "four": 4, "five": 5},
            stderr={"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5},
            ssize=35,
        )

        est_np = est.to_numpy()
        assert est.domains == list(est_np[:, 0])
        assert est.est == dict(zip(est.domains, est_np[:, 1]))
        assert est.stderr == dict(zip(est.domains, est_np[:, 2]))
        assert est.ssize == dict(zip(est.domains, est_np[:, 3]))

    def test_directest_polars(self):
        est = DirectEst(
            domain=["one", "two", "three", "four", "five"],
            est=1,  # {"one": 1, "two": 2},
            stderr=2,
            ssize=3,
        )

        est_pl = est.to_polars()
        assert (est.domains == est_pl.select("__domain").to_numpy().flatten()).all()
        assert est.est == dict(zip(est.domains, est_pl.select("est").to_numpy().flatten()))
        assert est.stderr == dict(zip(est.domains, est_pl.select("stderr").to_numpy().flatten()))
        assert est.ssize == dict(zip(est.domains, est_pl.select("ssize").to_numpy().flatten()))

    def test_directest_pandas(self):
        est = DirectEst(
            domain=["one", "two", "three", "four", "five"],
            est=[1, 2, 3, 4, 5],  # {"one": 1, "two": 2},
            stderr=[0.1, 0.2, 0.3, 0.4, 0.5],
            ssize={"one": 10, "two": 20, "three": 30, "four": 40, "five": 50},
        )

        est_pd = est.to_pandas()
        assert (est.domains == est_pd["__domain"].values).all()
        assert est.est == dict(zip(est.domains, est_pd["est"].values))
        assert est.stderr == dict(zip(est.domains, est_pd["stderr"].values))
        assert est.ssize == dict(zip(est.domains, est_pd["ssize"].values))


class TestDirectEst2:
    # Tests that an instance of DirectEst can be created with valid inputs.
    def test_create_instance_with_valid_inputs(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(est, stderr, ssize, psize, area)

        assert direct_est.domains == [1, 2, 3]
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

        direct_est = DirectEst(est, stderr, ssize, psize, area)

        cv = direct_est.cv

        assert cv == {1: 0.1, 2: 0.1, 3: 0.1}

    # Tests that an instance of DirectEst can be converted to a numpy array.
    def test_convert_to_numpy_array(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(est, stderr, ssize, psize, area)

        numpy_array = direct_est.to_numpy()

        assert numpy_array.shape == (3, 4)

    # Tests that an instance of DirectEst can be converted to a polars dataframe.
    def test_convert_to_polars_dataframe(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(est, stderr, ssize, psize, area)

        polars_df = direct_est.to_polars()

        assert polars_df.shape == (3, 4)

    # Tests that an instance of DirectEst can be converted to a pandas dataframe.
    def test_convert_to_pandas_dataframe(self):
        area = [1, 2, 3]
        est = {1: 10, 2: 20, 3: 30}
        stderr = {1: 1, 2: 2, 3: 3}
        ssize = {1: 100, 2: 200, 3: 300}
        psize = {1: 1000, 2: 2000, 3: 3000}

        direct_est = DirectEst(est, stderr, ssize, psize, area)

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

        direct_est = DirectEst(est, stderr, ssize, psize, area)

        assert direct_est.domains == [1]
        assert direct_est.est == {1: 10}
        assert direct_est.stderr == {1: 1}
        assert direct_est.ssize == {1: 100}
        assert direct_est.psize is None
