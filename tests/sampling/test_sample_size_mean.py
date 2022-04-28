import pytest

from samplics.sampling import SampleSize


size_str_mean_wald = SampleSize(parameter="mean", stratification=True)


@pytest.mark.xfail(strict=True, reason="Invalid method for proprotion")
def test_size_invalid_method_for_proportion():
    SampleSize(parameter="proportion", method="Whatever")


@pytest.mark.xfail(strict=True, reason="Invalid method for mean")
def test_size_invalid_method_for_mean():
    SampleSize(parameter="mean", method="fleiss")


# Wald's method
size_mean_nat = SampleSize(parameter="mean")


def test_size_nat_wald_basics():
    assert size_mean_nat.parameter == "mean"
    assert size_mean_nat.method == "wald"
    assert not size_mean_nat.stratification


@pytest.mark.xfail(strict=True, reason="Parameter target provided instead of sigma")
def test_size_nat_wald_should_fail():
    size_mean_nat.calculate(2, 1)


def test_size_nat_wald():
    size_mean_nat.calculate(half_ci=1, target=1, sigma=2)
    assert size_mean_nat.samp_size == 16
    assert size_mean_nat.deff_c == 1.0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2


def test_size_nat_wald_no_target():
    size_mean_nat.calculate(half_ci=1, sigma=2)
    assert size_mean_nat.samp_size == 16
    assert size_mean_nat.deff_c == 1.0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2


def test_size_nat_wald_deff0():
    size_mean_nat.calculate(half_ci=1, target=2, sigma=2, deff=0)
    assert size_mean_nat.samp_size == 0
    assert size_mean_nat.deff_c == 0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2


def test_size_nat_wald_deff1():
    size_mean_nat.calculate(half_ci=1, sigma=2, deff=1.3)
    assert size_mean_nat.samp_size == 20
    assert size_mean_nat.deff_c == 1.3
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2


def test_size_nat_wald_df1():
    size_mean_nat.calculate(
        half_ci=1,
        target=1,
        sigma=2,
    )
    size_df = size_mean_nat.to_dataframe()
    assert (size_df.columns == ["_parameter", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_nat_wald_df2():
    size_mean_nat.calculate(half_ci=1, target=1, sigma=2, deff=1.5)
    size_df = size_mean_nat.to_dataframe(["param", "mean", "sigma", "half_ci", "samp_size"])
    assert (size_df.columns == ["param", "mean", "sigma", "half_ci", "samp_size"]).all()


# Wald's method - stratified
size_str_mean_wald = SampleSize(parameter="mean", stratification=True)

half_ci = {"stratum1": 1, "stratum2": 1, "stratum3": 3}
deff = {"stratum1": 1, "stratum2": 1.3, "stratum3": 3}
sigma = {"stratum1": 2, "stratum2": 1, "stratum3": 5}


def test_size_mean_str_wald_basics():
    assert size_str_mean_wald.parameter == "mean"
    assert size_str_mean_wald.method == "wald"
    assert size_str_mean_wald.stratification


def test_size_mean_str_wald_size1():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=2)
    assert size_str_mean_wald.samp_size["stratum1"] == 16
    assert size_str_mean_wald.samp_size["stratum2"] == 16
    assert size_str_mean_wald.samp_size["stratum3"] == 2


def test_size_mean_str_wald_size2():
    size_str_mean_wald.calculate(half_ci=half_ci, target=2, sigma=2, deff=deff)
    assert size_str_mean_wald.samp_size["stratum1"] == 16
    assert size_str_mean_wald.samp_size["stratum2"] == 20
    assert size_str_mean_wald.samp_size["stratum3"] == 6


def test_size_str_mean_wald_size3():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=deff)
    assert size_str_mean_wald.samp_size["stratum1"] == 16
    assert size_str_mean_wald.samp_size["stratum2"] == 5
    assert size_str_mean_wald.samp_size["stratum3"] == 33


def test_size_str_mean_wald_size4():
    size_str_mean_wald.calculate(half_ci=half_ci, target=2, sigma=sigma, deff=2)
    assert size_str_mean_wald.samp_size["stratum1"] == 31
    assert size_str_mean_wald.samp_size["stratum2"] == 8
    assert size_str_mean_wald.samp_size["stratum3"] == 22


def test_size_mean_str_wald_df1():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=2)
    size_df = size_str_mean_wald.to_dataframe()
    assert size_df.shape[0] == 3
    assert (
        size_df.columns
        == ["_parameter", "_stratum", "_target", "_sigma", "_half_ci", "_samp_size"]
    ).all()


def test_size_mean_str_wald_df2():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=2)
    size_df = size_str_mean_wald.to_dataframe(["param", "str", "mean", "sigma", "E", "size"])
    assert size_df.shape[0] == 3
    assert (size_df.columns == ["param", "str", "mean", "sigma", "E", "size"]).all()


size_str_mean_wald_fpc = SampleSize(parameter="mean", stratification=True)


half_ci2 = {"stratum1": 0.5, "stratum2": 0.5, "stratum3": 0.5}
sigma2 = {"stratum1": 2, "stratum2": 2, "stratum3": 2}
pop_size2 = {"stratum1": 1000, "stratum2": 10000, "stratum3": 10000000}


def test_size_mean_str_wald_fpc1():
    size_str_mean_wald_fpc.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=pop_size2)
    assert size_str_mean_wald_fpc.samp_size["stratum1"] ==   58
    assert size_str_mean_wald_fpc.samp_size["stratum2"] == 62
    assert size_str_mean_wald_fpc.samp_size["stratum3"] == 62


def test_size_mean_str_wald_fpc2():
    size_str_mean_wald_fpc.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=1000)
    assert size_str_mean_wald_fpc.samp_size["stratum1"] == 62
    assert size_str_mean_wald_fpc.samp_size["stratum2"] == 62
    assert size_str_mean_wald_fpc.samp_size["stratum3"] == 62
