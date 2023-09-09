from samplics.sae.data_types import FitMethod, DirectEst, AuxVars

import numpy as np

# FitMethod


class Test_FitMethod:
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


class Test_DirectEst:
    def test_direct_est0(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est=1,  # {"one": 1, "two": 2},
            stderr=2,
            ssize=3,
        )
        assert est.area == ("one", "two", "three", "four", "five")
        assert est.est == {"one": 1, "two": 1, "three": 1, "four": 1, "five": 1}
        assert est.stderr == {"one": 2, "two": 2, "three": 2, "four": 2, "five": 2}
        assert est.ssize == {"one": 3, "two": 3, "three": 3, "four": 3, "five": 3}

    def test_direct_est1(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est=[1, 2, 3, 4, 5],  # {"one": 1, "two": 2},
            stderr=[0.1, 0.2, 0.3, 0.4, 0.5],
            ssize={"one": 10, "two": 20, "three": 30, "four": 40, "five": 50},
        )
        assert est.area == ("one", "two", "three", "four", "five")
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
        assert est.area == ("one", "two", "three", "four", "five")
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
        assert est.area == ("one", "two", "three", "four", "five")
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
        assert est.area == tuple(est_np[:, 0])
        assert est.est == dict(zip(est.area, est_np[:, 1]))
        assert est.stderr == dict(zip(est.area, est_np[:, 2]))
        assert est.ssize == dict(zip(est.area, est_np[:, 3]))

    def test_directest_polars(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est=1,  # {"one": 1, "two": 2},
            stderr=2,
            ssize=3,
        )

        est_pl = est.to_polars()
        assert (est.area == est_pl.select("area").to_numpy().flatten()).all()
        assert est.est == dict(zip(est.area, est_pl.select("est").to_numpy().flatten()))
        assert est.stderr == dict(zip(est.area, est_pl.select("stderr").to_numpy().flatten()))
        assert est.ssize == dict(zip(est.area, est_pl.select("ssize").to_numpy().flatten()))

    def test_directest_pandas(self):
        est = DirectEst(
            area=["one", "two", "three", "four", "five"],
            est=[1, 2, 3, 4, 5],  # {"one": 1, "two": 2},
            stderr=[0.1, 0.2, 0.3, 0.4, 0.5],
            ssize={"one": 10, "two": 20, "three": 30, "four": 40, "five": 50},
        )

        est_pd = est.to_pandas()
        assert (est.area == est_pd["area"].values).all()
        assert est.est == dict(zip(est.area, est_pd["est"].values))
        assert est.stderr == dict(zip(est.area, est_pd["stderr"].values))
        assert est.ssize == dict(zip(est.area, est_pd["ssize"].values))


# Auxiliary variables


class AuxVars:
    aux_vars1 = AuxVars(
        record_id=None,  # (11, 2, 22, 3, 13),
        area=("one", "two", "three", "one", "three"),
        sex=["male", "female", "male", "male", "female"],
        age=[23, 12, 3, 2, 47],
    )

    aux_vars1.to_numpy()
