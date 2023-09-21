from samplics.types.options import FitMethod


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
