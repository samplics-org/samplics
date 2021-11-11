import pytest

from samplics.sampling import SampleSize


size_mean_nat = SampleSize(parameter="mean")


def test_size_nat_wald_basics():
    assert size_mean_nat.parameter == "mean"
    assert size_mean_nat.method == "wald"
    assert not size_mean_nat.stratification


@pytest.mark.xfail(strict=True, reason="Parameter target provided instead of sigma")
def test_size_nat_wald_should_fail():
    size_mean_nat.calculate(2, 1)


def test_size_nat_wald():
    size_mean_nat.calculate(half_ci=1, sigma=2)
    assert size_mean_nat.samp_size == 16
    assert size_mean_nat.deff_c == 1.0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2


def test_size_nat_wald_deff0():
    size_mean_nat.calculate(half_ci=1, sigma=2, deff=0)
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
        sigma=2,
    )
    size_df = size_mean_nat.to_dataframe()
    assert (size_df.columns == ["_parameter", "_target", "_half_ci", "_samp_size"]).all()


def test_size_nat_wald_df2():
    size_mean_nat.calculate(half_ci=1, sigma=2, deff=1.5)
    size_df = size_mean_nat.to_dataframe()
    assert (size_df.columns == ["_parameter", "_target", "_half_ci", "_samp_size"]).all()
