import pytest

from samplics.sampling import SampleSize
from samplics.utils import PopParam, SizeMethod


@pytest.mark.xfail(strict=True, reason="Invalid method for proprotion")
def test_size_invalid_method_for_prop():
    SampleSize(param=PopParam.prop, method="Whatever")


@pytest.mark.xfail(strict=True, reason="Invalid method for mean")
def test_size_invalid_method_for_mean():
    SampleSize(param=PopParam.mean, method=SizeMethod.fleiss)


# NOT-STRATIFIED Wald's method
size_mean_nat = SampleSize(param=PopParam.mean)


def test_size_nat_wald_basics():
    assert size_mean_nat.param == PopParam.mean
    assert size_mean_nat.method == SizeMethod.wald
    assert not size_mean_nat.strat


@pytest.mark.xfail(strict=True, reason="param target provided instead of sigma")
def test_size_nat_wald_should_fail():
    size_mean_nat.calculate(2, 1)


def test_size_nat_wald():
    size_mean_nat.calculate(half_ci=1, target=1, sigma=2)
    assert size_mean_nat.samp_size == 16
    assert size_mean_nat.deff_c == 1.0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2
    assert size_mean_nat.resp_rate == 1


def test_size_nat_with_resp_rate_wald():
    size_mean_nat.calculate(half_ci=1, target=1, sigma=2, resp_rate=0.8)
    assert size_mean_nat.samp_size == 20
    assert size_mean_nat.deff_c == 1.0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2
    assert size_mean_nat.resp_rate == 0.8


def test_size_nat_wald_no_target_wald():
    size_mean_nat.calculate(half_ci=1, sigma=2)
    assert size_mean_nat.samp_size == 16
    assert size_mean_nat.deff_c == 1.0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2
    assert size_mean_nat.resp_rate == 1


def test_size_nat_wald_no_target_with_resp_rate():
    size_mean_nat.calculate(half_ci=1, sigma=2, resp_rate=0.9)
    assert size_mean_nat.samp_size == 18
    assert size_mean_nat.deff_c == 1.0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2


def test_size_nat_wald_deff0():
    size_mean_nat.calculate(half_ci=1, target=2, sigma=2, deff=0)
    assert size_mean_nat.samp_size == 0
    assert size_mean_nat.deff_c == 0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2
    assert size_mean_nat.resp_rate == 1


def test_size_nat_wald_deff0_with_resp_rate():
    size_mean_nat.calculate(half_ci=1, target=2, sigma=2, deff=0, resp_rate=0.9)
    assert size_mean_nat.samp_size == 0
    assert size_mean_nat.deff_c == 0
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2
    assert size_mean_nat.resp_rate == 0.9


def test_size_nat_wald_deff1():
    size_mean_nat.calculate(half_ci=1, sigma=2, deff=1.3)
    assert size_mean_nat.samp_size == 20
    assert size_mean_nat.deff_c == 1.3
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2
    assert size_mean_nat.resp_rate == 1


def test_size_nat_wald_deff1_with_resp_rate():
    size_mean_nat.calculate(half_ci=1, sigma=2, deff=1.3, resp_rate=0.75)
    assert size_mean_nat.samp_size == 27
    assert size_mean_nat.deff_c == 1.3
    assert size_mean_nat.half_ci == 1
    assert size_mean_nat.sigma == 2
    assert size_mean_nat.resp_rate == 0.75


@pytest.mark.xfail(strict=True, reason="Response rate is 0")
def test_size_nat_wald_deff2_with_resp_rate():
    size_mean_nat.calculate(half_ci=1, sigma=2, deff=1.3, resp_rate=0)
    assert size_mean_nat.samp_size == 20


def test_size_nat_wald_df3():
    size_mean_nat.calculate(
        half_ci=1,
        target=1,
        sigma=2,
    )
    size_df = size_mean_nat.to_dataframe()
    assert (size_df.columns == ["_param", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_nat_wald_df3_with_resp_rate():
    size_mean_nat.calculate(half_ci=1, target=1, sigma=2, resp_rate=0.3)
    size_df = size_mean_nat.to_dataframe()
    assert (size_df.columns == ["_param", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_nat_wald_df4():
    size_mean_nat.calculate(half_ci=1, target=1, sigma=2, deff=1.5)
    size_df = size_mean_nat.to_dataframe(["param", "mean", "sigma", "half_ci", "samp_size"])
    assert (size_df.columns == ["param", "mean", "sigma", "half_ci", "samp_size"]).all()


# STRATIFIED Wald's method
size_str_mean_wald = SampleSize(param=PopParam.mean, strat=True)

half_ci = {"stratum1": 1, "stratum2": 1, "stratum3": 3}
deff = {"stratum1": 1, "stratum2": 1.3, "stratum3": 3}
sigma = {"stratum1": 2, "stratum2": 1, "stratum3": 5}
resp_rate = {"stratum1": 0.8, "stratum2": 0.3, "stratum3": 0.5}


def test_size_mean_str_wald_basics():
    assert size_str_mean_wald.param == PopParam.mean
    assert size_str_mean_wald.method == SizeMethod.wald
    assert size_str_mean_wald.strat


def test_size_mean_str_wald_size1():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=2)
    assert size_str_mean_wald.samp_size["stratum1"] == 16
    assert size_str_mean_wald.samp_size["stratum2"] == 16
    assert size_str_mean_wald.samp_size["stratum3"] == 2
    assert size_str_mean_wald.resp_rate["stratum1"] == 1
    assert size_str_mean_wald.resp_rate["stratum2"] == 1
    assert size_str_mean_wald.resp_rate["stratum3"] == 1


def test_size_mean_str_wald_size1_with_resp_rate1():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=2, resp_rate=0.8)
    assert size_str_mean_wald.samp_size["stratum1"] == 20
    assert size_str_mean_wald.samp_size["stratum2"] == 20
    assert size_str_mean_wald.samp_size["stratum3"] == 3
    assert size_str_mean_wald.resp_rate["stratum1"] == 0.8
    assert size_str_mean_wald.resp_rate["stratum2"] == 0.8
    assert size_str_mean_wald.resp_rate["stratum3"] == 0.8


def test_size_mean_str_wald_size1_with_resp_rate2():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=2, resp_rate=resp_rate)
    assert size_str_mean_wald.samp_size["stratum1"] == 20
    assert size_str_mean_wald.samp_size["stratum2"] == 52
    assert size_str_mean_wald.samp_size["stratum3"] == 4
    assert size_str_mean_wald.resp_rate["stratum1"] == 0.8
    assert size_str_mean_wald.resp_rate["stratum2"] == 0.3
    assert size_str_mean_wald.resp_rate["stratum3"] == 0.5


def test_size_mean_str_wald_size2():
    size_str_mean_wald.calculate(half_ci=half_ci, target=2, sigma=2, deff=deff)
    assert size_str_mean_wald.samp_size["stratum1"] == 16
    assert size_str_mean_wald.samp_size["stratum2"] == 20
    assert size_str_mean_wald.samp_size["stratum3"] == 6
    assert size_str_mean_wald.resp_rate["stratum1"] == 1.0
    assert size_str_mean_wald.resp_rate["stratum2"] == 1.0
    assert size_str_mean_wald.resp_rate["stratum3"] == 1.0


def test_size_mean_str_wald_size2_with_resp_rate1():
    size_str_mean_wald.calculate(half_ci=half_ci, target=2, sigma=2, deff=deff, resp_rate=0.5)
    assert size_str_mean_wald.samp_size["stratum1"] == 31
    assert size_str_mean_wald.samp_size["stratum2"] == 40
    assert size_str_mean_wald.samp_size["stratum3"] == 11
    assert size_str_mean_wald.resp_rate["stratum1"] == 0.5
    assert size_str_mean_wald.resp_rate["stratum2"] == 0.5
    assert size_str_mean_wald.resp_rate["stratum3"] == 0.5


def test_size_mean_str_wald_size2_with_resp_rate2():
    size_str_mean_wald.calculate(half_ci=half_ci, target=2, sigma=2, deff=deff, resp_rate=resp_rate)
    assert size_str_mean_wald.samp_size["stratum1"] == 20
    assert size_str_mean_wald.samp_size["stratum2"] == 67
    assert size_str_mean_wald.samp_size["stratum3"] == 11
    assert size_str_mean_wald.resp_rate["stratum1"] == 0.8
    assert size_str_mean_wald.resp_rate["stratum2"] == 0.3
    assert size_str_mean_wald.resp_rate["stratum3"] == 0.5


def test_size_str_mean_wald_size3():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=deff)
    assert size_str_mean_wald.samp_size["stratum1"] == 16
    assert size_str_mean_wald.samp_size["stratum2"] == 5
    assert size_str_mean_wald.samp_size["stratum3"] == 33
    assert size_str_mean_wald.resp_rate["stratum1"] == 1
    assert size_str_mean_wald.resp_rate["stratum2"] == 1
    assert size_str_mean_wald.resp_rate["stratum3"] == 1


def test_size_str_mean_wald_size3_with_resp_rate1():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=deff, resp_rate=0.7)
    assert size_str_mean_wald.samp_size["stratum1"] == 22
    assert size_str_mean_wald.samp_size["stratum2"] == 8
    assert size_str_mean_wald.samp_size["stratum3"] == 46
    assert size_str_mean_wald.resp_rate["stratum1"] == 0.7
    assert size_str_mean_wald.resp_rate["stratum2"] == 0.7
    assert size_str_mean_wald.resp_rate["stratum3"] == 0.7


def test_size_str_mean_wald_size3_with_resp_rate2():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=deff, resp_rate=resp_rate)
    assert size_str_mean_wald.samp_size["stratum1"] == 20
    assert size_str_mean_wald.samp_size["stratum2"] == 17
    assert size_str_mean_wald.samp_size["stratum3"] == 65
    assert size_str_mean_wald.resp_rate["stratum1"] == 0.8
    assert size_str_mean_wald.resp_rate["stratum2"] == 0.3
    assert size_str_mean_wald.resp_rate["stratum3"] == 0.5


def test_size_str_mean_wald_size4():
    size_str_mean_wald.calculate(half_ci=half_ci, target=2, sigma=sigma, deff=2)
    assert size_str_mean_wald.samp_size["stratum1"] == 31
    assert size_str_mean_wald.samp_size["stratum2"] == 8
    assert size_str_mean_wald.samp_size["stratum3"] == 22


def test_size_str_mean_wald_size4_with_resp_rate():
    size_str_mean_wald.calculate(half_ci=half_ci, target=2, sigma=sigma, deff=2, resp_rate=0.6)
    assert size_str_mean_wald.samp_size["stratum1"] == 52
    assert size_str_mean_wald.samp_size["stratum2"] == 13
    assert size_str_mean_wald.samp_size["stratum3"] == 36


def test_size_mean_str_wald_df1():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=2)
    size_df = size_str_mean_wald.to_dataframe()
    assert size_df.shape[0] == 3
    assert (size_df.columns == ["_param", "_stratum", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_mean_str_wald_df1_with_resp_rate():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=2, resp_rate=1)
    size_df = size_str_mean_wald.to_dataframe()
    assert size_df.shape[0] == 3
    assert (size_df.columns == ["_param", "_stratum", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_mean_str_wald_df2():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=2)
    size_df = size_str_mean_wald.to_dataframe(["param", "str", "mean", "sigma", "E", "size"])
    assert size_df.shape[0] == 3
    assert (size_df.columns == ["param", "str", "mean", "sigma", "E", "size"]).all()


def test_size_mean_str_wald_df2_with_resp_rate():
    size_str_mean_wald.calculate(half_ci=half_ci, target=1, sigma=sigma, deff=2, resp_rate=0.65)
    size_df = size_str_mean_wald.to_dataframe(["param", "str", "mean", "sigma", "E", "size"])
    assert size_df.shape[0] == 3
    assert (size_df.columns == ["param", "str", "mean", "sigma", "E", "size"]).all()


size_str_mean_wald_fpc = SampleSize(param=PopParam.mean, strat=True)


half_ci2 = {"stratum1": 0.5, "stratum2": 0.5, "stratum3": 0.5}
sigma2 = {"stratum1": 2, "stratum2": 2, "stratum3": 2}
pop_size2 = {"stratum1": 1000, "stratum2": 10000, "stratum3": 10000000}
resp_rate2 = {"stratum1": 0.4, "stratum2": 1, "stratum3": 0.65}


def test_size_mean_str_wald_fpc1():
    size_str_mean_wald_fpc.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=pop_size2)
    assert size_str_mean_wald_fpc.samp_size["stratum1"] == 58
    assert size_str_mean_wald_fpc.samp_size["stratum2"] == 62
    assert size_str_mean_wald_fpc.samp_size["stratum3"] == 62


def test_size_mean_str_wald_fpc1_with_resp_rate1():
    size_str_mean_wald_fpc.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=pop_size2, resp_rate=0.5)
    assert size_str_mean_wald_fpc.samp_size["stratum1"] == 116
    assert size_str_mean_wald_fpc.samp_size["stratum2"] == 123
    assert size_str_mean_wald_fpc.samp_size["stratum3"] == 123


def test_size_mean_str_wald_fpc1_with_resp_rate2():
    size_str_mean_wald_fpc.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=pop_size2, resp_rate=resp_rate2)
    assert size_str_mean_wald_fpc.samp_size["stratum1"] == 145
    assert size_str_mean_wald_fpc.samp_size["stratum2"] == 62
    assert size_str_mean_wald_fpc.samp_size["stratum3"] == 95


def test_size_mean_str_wald_fpc2():
    size_str_mean_wald_fpc.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=1000)
    assert size_str_mean_wald_fpc.samp_size["stratum1"] == 58
    assert size_str_mean_wald_fpc.samp_size["stratum2"] == 58
    assert size_str_mean_wald_fpc.samp_size["stratum3"] == 58


def test_size_mean_str_wald_fpc2_with_resp_rate1():
    size_str_mean_wald_fpc.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=1000, resp_rate=0.45)
    assert size_str_mean_wald_fpc.samp_size["stratum1"] == 129
    assert size_str_mean_wald_fpc.samp_size["stratum2"] == 129
    assert size_str_mean_wald_fpc.samp_size["stratum3"] == 129


def test_size_mean_str_wald_fpc2_with_resp_rate2():
    size_str_mean_wald_fpc.calculate(half_ci=half_ci2, sigma=sigma2, pop_size=1000, resp_rate=resp_rate2)
    assert size_str_mean_wald_fpc.samp_size["stratum1"] == 145
    assert size_str_mean_wald_fpc.samp_size["stratum2"] == 58
    assert size_str_mean_wald_fpc.samp_size["stratum3"] == 90
