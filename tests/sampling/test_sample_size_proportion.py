import pytest

from samplics.sampling import SampleSize, allocate
from samplics.utils import PopParam, SizeMethod


region = ["Dakar", "Kaolack", "Ziguinchor"]
pop_size = {"Dakar": 500, "Kaolack": 300, "Ziguinchor": 200}


@pytest.mark.xfail(reason="stratum is required")
def test_equal_stratum_error():
    allocate(method="equal")


def test_equal():
    sizes, rates = allocate(method="equal", stratum=region, pop_size=pop_size, constant=15)
    assert sizes["Dakar"] == 15
    assert sizes["Kaolack"] == 15
    assert sizes["Ziguinchor"] == 15
    assert rates["Dakar"] == 15 / pop_size["Dakar"]
    assert rates["Kaolack"] == 15 / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == 15 / pop_size["Ziguinchor"]
    assert rates["Dakar"] == sizes["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == sizes["Kaolack"] / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == sizes["Ziguinchor"] / pop_size["Ziguinchor"]


def test_equal_error():
    with pytest.raises(ValueError):
        allocate(method="equal", stratum=region, pop_size=pop_size, constant=[23])


def test_propal():
    sizes, rates = allocate(method="propal", stratum=region, samp_size=100, pop_size=pop_size)
    assert sizes["Dakar"] == 50
    assert sizes["Kaolack"] == 30
    assert sizes["Ziguinchor"] == 20
    assert rates["Dakar"] == sizes["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == sizes["Kaolack"] / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == sizes["Ziguinchor"] / pop_size["Ziguinchor"]


def test_propal_error():
    with pytest.raises(ValueError):
        allocate(method="equal", stratum=region, samp_size=100, pop_size=5)


def test_fixed_rate_number():
    sizes, rates = allocate(method="fixed_rate", stratum=region, pop_size=pop_size, rate=0.05)
    assert sizes["Dakar"] == 25
    assert sizes["Kaolack"] == 15
    assert sizes["Ziguinchor"] == 10
    assert rates["Dakar"] == 0.05
    assert rates["Kaolack"] == 0.05
    assert rates["Ziguinchor"] == 0.05
    assert rates["Dakar"] == sizes["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == sizes["Kaolack"] / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == sizes["Ziguinchor"] / pop_size["Ziguinchor"]


def test_fixed_rate_error():
    with pytest.raises(ValueError):
        allocate(method="fixed_rate", stratum=region, pop_size=[5])


def test_variable_rate():
    input_rates = {"Dakar": 0.05, "Kaolack": 0.10, "Ziguinchor": 0.20}
    sizes, rates = allocate(
        method="variable_rate", stratum=region, pop_size=pop_size, rate=input_rates
    )
    assert sizes["Dakar"] == 25
    assert sizes["Kaolack"] == 30
    assert sizes["Ziguinchor"] == 40
    assert rates["Dakar"] == 0.05
    assert rates["Kaolack"] == 0.10
    assert rates["Ziguinchor"] == 0.20
    assert rates["Dakar"] == sizes["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == sizes["Kaolack"] / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == sizes["Ziguinchor"] / pop_size["Ziguinchor"]


def testf_variable_rate_error():
    with pytest.raises(ValueError):
        allocate(method="variable_rate", stratum=region, pop_size=[5])


def test_propal_rate():
    pop_size2 = {"Dakar": 5000, "Kaolack": 3000, "Ziguinchor": 2000}
    sizes, rates = allocate(
        method="propal_rate", stratum=region, pop_size=pop_size2, rate=0.000005
    )
    assert sizes["Dakar"] == 125
    assert sizes["Kaolack"] == 45
    assert sizes["Ziguinchor"] == 20
    assert rates["Dakar"] == pytest.approx(0.000005 * pop_size2["Dakar"])
    assert rates["Kaolack"] == pytest.approx(0.000005 * pop_size2["Kaolack"])
    assert rates["Ziguinchor"] == pytest.approx(0.000005 * pop_size2["Ziguinchor"])
    assert rates["Dakar"] == sizes["Dakar"] / pop_size2["Dakar"]
    assert rates["Kaolack"] == sizes["Kaolack"] / pop_size2["Kaolack"]
    assert rates["Ziguinchor"] == sizes["Ziguinchor"] / pop_size2["Ziguinchor"]


def test_propal_rate_error1():
    rate = {"Dakar": 0.005, "Kaolack": 0.010, "Ziguinchor": 0.020}
    with pytest.raises(ValueError):
        allocate(method="propal_rate", stratum=region, pop_size=pop_size, rate=rate)


def test_deff_propal_rate_error2():
    with pytest.raises(ValueError):
        allocate(method="fixed_rate", stratum=region, pop_size=[5])


def test_deff_optimum_mean():
    stddev = {"Dakar": 5, "Kaolack": 10, "Ziguinchor": 20}
    sizes, rates = allocate(
        method="optimum_mean", stratum=region, pop_size=pop_size, rate=0.01, stddev=stddev
    )
    assert sizes["Dakar"] == 25
    assert sizes["Kaolack"] == 30
    assert sizes["Ziguinchor"] == 40
    assert rates["Dakar"] == 0.01 * stddev["Dakar"]
    assert rates["Kaolack"] == 0.01 * stddev["Kaolack"]
    assert rates["Ziguinchor"] == 0.01 * stddev["Ziguinchor"]
    assert rates["Dakar"] == sizes["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == sizes["Kaolack"] / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == sizes["Ziguinchor"] / pop_size["Ziguinchor"]


def test_optimum_mean_error1():
    rate = {"Dakar": 0.005, "Kaolack": 0.010, "Ziguinchor": 0.020}
    stddev = {"Dakar": 5, "Kaolack": 10, "Ziguinchor": 20}
    with pytest.raises(ValueError):
        allocate(
            method="optimum_mean", stratum=region, pop_size=pop_size, rate=rate, stddev=stddev
        )


def test_optimum_mean_error2():
    with pytest.raises(ValueError):
        allocate(method="optimum_mean", stratum=region, pop_size=pop_size, stddev=[5])


def test_deff_optimum_comparison():
    stddev = {"Dakar": 50, "Kaolack": 10, "Ziguinchor": 20}
    sizes, rates = allocate(
        method="optimum_comparison", stratum=region, pop_size=pop_size, rate=0.5, stddev=stddev
    )
    assert sizes["Dakar"] == 25
    assert sizes["Kaolack"] == 5
    assert sizes["Ziguinchor"] == 10
    assert rates["Dakar"] == 0.5 * stddev["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == 0.5 * stddev["Kaolack"] / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == 0.5 * stddev["Ziguinchor"] / pop_size["Ziguinchor"]
    assert rates["Dakar"] == sizes["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == sizes["Kaolack"] / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == sizes["Ziguinchor"] / pop_size["Ziguinchor"]


def test_optimum_comparison_error1():
    rate = {"Dakar": 0.005, "Kaolack": 0.010, "Ziguinchor": 0.020}
    stddev = {"Dakar": 5, "Kaolack": 10, "Ziguinchor": 20}
    with pytest.raises(ValueError):
        allocate(
            method="optimum_comparison",
            stratum=region,
            pop_size=pop_size,
            rate=rate,
            stddev=stddev,
        )


def test_equal_errors_error2():
    with pytest.raises(ValueError):
        allocate(method="equal_errors", stratum=region, pop_size=pop_size, stddev=[5])


def test_deff_equal_errors():
    stddev = {"Dakar": 5, "Kaolack": 1, "Ziguinchor": 2}
    sizes, rates = allocate(
        method="equal_errors", stratum=region, pop_size=pop_size, constant=5, stddev=stddev
    )
    assert sizes["Dakar"] == 125
    assert sizes["Kaolack"] == 5
    assert sizes["Ziguinchor"] == 20
    assert rates["Dakar"] == 5 * stddev["Dakar"] * stddev["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == 5 * stddev["Kaolack"] * stddev["Kaolack"] / pop_size["Kaolack"]
    assert (
        rates["Ziguinchor"]
        == 5 * stddev["Ziguinchor"] * stddev["Ziguinchor"] / pop_size["Ziguinchor"]
    )
    assert rates["Dakar"] == sizes["Dakar"] / pop_size["Dakar"]
    assert rates["Kaolack"] == sizes["Kaolack"] / pop_size["Kaolack"]
    assert rates["Ziguinchor"] == sizes["Ziguinchor"] / pop_size["Ziguinchor"]


def test_equal_errors_error1():
    stddev = {"Dakar": 5, "Kaolack": 10, "Ziguinchor": 20}
    with pytest.raises(ValueError):
        allocate(
            method="equal_errors",
            stratum=region,
            pop_size=pop_size,
            constant=[9],
            stddev=stddev,
        )


def test_optimum_comparison_error2():
    with pytest.raises(ValueError):
        allocate(method="optimum_comparison", stratum=region, pop_size=pop_size, stddev=[5])


## Design effects


deff_calculation = SampleSize()


def test_deff_int():
    assert deff_calculation.deff(cluster_size=30, icc=0.01) == 1.29


def test_deff_float():
    assert deff_calculation.deff(15.5, 0.03) == 1.435


def test_deff_dict():
    m = {"stratum1": 30, "stratum2": 15.5, "stratum3": 50}
    icc = {"stratum1": 0.01, "stratum2": 0.03, "stratum3": 0.10}
    deff = deff_calculation.deff(m, icc)
    assert deff == {"stratum1": 1.29, "stratum2": 1.435, "stratum3": 5.9}


## Wald's method - NOT-STRATIFIED

size_nat_wald = SampleSize()


def test_size_nat_wald_basics():
    assert size_nat_wald.param == PopParam.prop
    assert size_nat_wald.method == SizeMethod.wald
    assert not size_nat_wald.strat


def test_size_nat_wald_size():
    size_nat_wald.calculate(target=0.80, half_ci=0.10)
    assert size_nat_wald.samp_size == 62
    assert size_nat_wald.deff_c == 1.0
    assert size_nat_wald.target == 0.80
    assert size_nat_wald.half_ci == 0.1
    assert size_nat_wald.resp_rate == 1.0


def test_size_nat_wald_size_with_resp_rate():
    size_nat_wald.calculate(target=0.80, half_ci=0.10, resp_rate=0.5)
    assert size_nat_wald.samp_size == 123
    assert size_nat_wald.deff_c == 1.0
    assert size_nat_wald.target == 0.80
    assert size_nat_wald.half_ci == 0.1
    assert size_nat_wald.resp_rate == 0.5


def test_size_nat_wald_size_with_deff():
    size_nat_wald.calculate(target=0.80, half_ci=0.10, deff=1.5)
    assert size_nat_wald.samp_size == 93
    assert size_nat_wald.deff_c == 1.5
    assert size_nat_wald.target == 0.80
    assert size_nat_wald.half_ci == 0.1
    assert size_nat_wald.resp_rate == 1.0


def test_size_nat_wald_size_with_deff_and_resp_rate():
    size_nat_wald.calculate(target=0.80, half_ci=0.10, deff=1.5, resp_rate=0.7)
    assert size_nat_wald.samp_size == 132
    assert size_nat_wald.deff_c == 1.5
    assert size_nat_wald.target == 0.80
    assert size_nat_wald.half_ci == 0.1
    assert size_nat_wald.resp_rate == 0.7


def test_size_nat_wald_df():
    size_nat_wald.calculate(target=0.80, half_ci=0.10)
    size_df = size_nat_wald.to_dataframe()
    assert (size_df.columns == ["_param", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_nat_wald_df_with_resp_rate():
    size_nat_wald.calculate(target=0.80, half_ci=0.10, resp_rate=0.5)
    size_df = size_nat_wald.to_dataframe()
    assert (size_df.columns == ["_param", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_nat_wald_pop_size1():
    size_nat_wald.calculate(target=0.50, half_ci=0.03)
    assert size_nat_wald.samp_size == 1068


def test_size_nat_wald_pop_size1_with_resp_rate():
    size_nat_wald.calculate(target=0.50, half_ci=0.03, resp_rate=0.9)
    assert size_nat_wald.samp_size == 1186


def test_size_nat_wald_pop_size2():
    size_nat_wald.calculate(half_ci=0.03, target=0.50, pop_size=1251)
    assert size_nat_wald.samp_size == 577


def test_size_nat_wald_pop_size2_with_resp_rate():
    size_nat_wald.calculate(half_ci=0.03, target=0.50, pop_size=1251, resp_rate=0.6)
    assert size_nat_wald.samp_size == 961


##  Wald's method - STRATIFIED
size_str_wald = SampleSize(param=PopParam.prop, method=SizeMethod.wald, strat=True)

target = {"stratum1": 0.95, "stratum2": 0.70, "stratum3": 0.30}
half_ci = {"stratum1": 0.30, "stratum2": 0.10, "stratum3": 0.15}
deff = {"stratum1": 1, "stratum2": 1.5, "stratum3": 2.5}
resp_rate = {"stratum1": 0.95, "stratum2": 0.70, "stratum3": 0.30}


def test_size_str_wald_basics():
    assert size_str_wald.param == PopParam.prop
    assert size_str_wald.method == SizeMethod.wald
    assert size_str_wald.strat


def test_size_str_wald_size1():
    size_str_wald.calculate(target=target, half_ci=0.10)
    assert size_str_wald.samp_size["stratum1"] == 19
    assert size_str_wald.samp_size["stratum2"] == 81
    assert size_str_wald.samp_size["stratum3"] == 81


def test_size_str_wald_size1_with_resp_rate1():
    size_str_wald.calculate(target=target, half_ci=0.10, resp_rate=0.5)
    assert size_str_wald.samp_size["stratum1"] == 37
    assert size_str_wald.samp_size["stratum2"] == 162
    assert size_str_wald.samp_size["stratum3"] == 162
    assert size_str_wald.resp_rate["stratum1"] == 0.5
    assert size_str_wald.resp_rate["stratum2"] == 0.5
    assert size_str_wald.resp_rate["stratum3"] == 0.5


def test_size_str_wald_size1_with_resp_rate2():
    size_str_wald.calculate(target=target, half_ci=0.10, resp_rate=resp_rate)
    assert size_str_wald.samp_size["stratum1"] == 20
    assert size_str_wald.samp_size["stratum2"] == 116
    assert size_str_wald.samp_size["stratum3"] == 269
    assert size_str_wald.resp_rate["stratum1"] == 0.95
    assert size_str_wald.resp_rate["stratum2"] == 0.70
    assert size_str_wald.resp_rate["stratum3"] == 0.30


def test_size_str_wald_size2():
    size_str_wald.calculate(target=0.8, half_ci=half_ci)
    assert size_str_wald.samp_size["stratum1"] == 7
    assert size_str_wald.samp_size["stratum2"] == 62
    assert size_str_wald.samp_size["stratum3"] == 28


def test_size_str_wald_size2_with_resp_rate1():
    size_str_wald.calculate(target=0.8, half_ci=half_ci, resp_rate=0.8)
    assert size_str_wald.samp_size["stratum1"] == 9
    assert size_str_wald.samp_size["stratum2"] == 77
    assert size_str_wald.samp_size["stratum3"] == 35
    assert size_str_wald.resp_rate["stratum1"] == 0.8
    assert size_str_wald.resp_rate["stratum2"] == 0.8
    assert size_str_wald.resp_rate["stratum3"] == 0.8


def test_size_str_wald_size2_with_resp_rate2():
    size_str_wald.calculate(target=0.8, half_ci=half_ci, resp_rate=resp_rate)
    assert size_str_wald.samp_size["stratum1"] == 8
    assert size_str_wald.samp_size["stratum2"] == 88
    assert size_str_wald.samp_size["stratum3"] == 92
    assert size_str_wald.resp_rate["stratum1"] == 0.95
    assert size_str_wald.resp_rate["stratum2"] == 0.70
    assert size_str_wald.resp_rate["stratum3"] == 0.30


def test_size_str_wald_size3():
    size_str_wald.calculate(target=0.8, half_ci=0.10, deff=deff)
    assert size_str_wald.samp_size["stratum1"] == 62
    assert size_str_wald.samp_size["stratum2"] == 93
    assert size_str_wald.samp_size["stratum3"] == 154


def test_size_str_wald_size3_with_resp_rate1():
    size_str_wald.calculate(target=0.8, half_ci=0.10, deff=deff, resp_rate=0.6)
    assert size_str_wald.samp_size["stratum1"] == 103
    assert size_str_wald.samp_size["stratum2"] == 154
    assert size_str_wald.samp_size["stratum3"] == 257
    assert size_str_wald.resp_rate["stratum1"] == 0.6
    assert size_str_wald.resp_rate["stratum2"] == 0.6
    assert size_str_wald.resp_rate["stratum3"] == 0.6


def test_size_str_wald_size3_with_resp_rate2():
    size_str_wald.calculate(target=0.8, half_ci=0.10, deff=deff, resp_rate=resp_rate)
    assert size_str_wald.samp_size["stratum1"] == 65
    assert size_str_wald.samp_size["stratum2"] == 132
    assert size_str_wald.samp_size["stratum3"] == 513
    assert size_str_wald.resp_rate["stratum1"] == 0.95
    assert size_str_wald.resp_rate["stratum2"] == 0.70
    assert size_str_wald.resp_rate["stratum3"] == 0.30


def test_size_str_wald_size4():
    size_str_wald.calculate(target=target, half_ci=half_ci, deff=deff)
    assert size_str_wald.samp_size["stratum1"] == 3
    assert size_str_wald.samp_size["stratum2"] == 122
    assert size_str_wald.samp_size["stratum3"] == 90


def test_size_str_wald_size4_with_resp_rate1():
    size_str_wald.calculate(target=target, half_ci=half_ci, deff=deff, resp_rate=0.3)
    assert size_str_wald.samp_size["stratum1"] == 7
    assert size_str_wald.samp_size["stratum2"] == 404
    assert size_str_wald.samp_size["stratum3"] == 299
    assert size_str_wald.resp_rate["stratum1"] == 0.3
    assert size_str_wald.resp_rate["stratum2"] == 0.3
    assert size_str_wald.resp_rate["stratum3"] == 0.3


def test_size_str_wald_size4_with_resp_rate2():
    size_str_wald.calculate(target=target, half_ci=half_ci, deff=deff, resp_rate=resp_rate)
    assert size_str_wald.samp_size["stratum1"] == 3
    assert size_str_wald.samp_size["stratum2"] == 173
    assert size_str_wald.samp_size["stratum3"] == 299
    assert size_str_wald.resp_rate["stratum1"] == 0.95
    assert size_str_wald.resp_rate["stratum2"] == 0.7
    assert size_str_wald.resp_rate["stratum3"] == 0.3


def test_size_str_wald_size5():
    size_str_wald.calculate(target=0.8, half_ci=0.1, deff=1.5, number_strata=5)
    assert size_str_wald.samp_size["_stratum_1"] == 93
    assert size_str_wald.samp_size["_stratum_2"] == 93
    assert size_str_wald.samp_size["_stratum_3"] == 93
    assert size_str_wald.samp_size["_stratum_4"] == 93
    assert size_str_wald.samp_size["_stratum_5"] == 93


def test_size_str_wald_size5_with_resp_rate():
    size_str_wald.calculate(target=0.8, half_ci=0.1, deff=1.5, number_strata=5, resp_rate=0.5)
    assert size_str_wald.samp_size["_stratum_1"] == 185
    assert size_str_wald.samp_size["_stratum_2"] == 185
    assert size_str_wald.samp_size["_stratum_3"] == 185
    assert size_str_wald.samp_size["_stratum_4"] == 185
    assert size_str_wald.samp_size["_stratum_5"] == 185
    assert size_str_wald.resp_rate["_stratum_1"] == 0.5
    assert size_str_wald.resp_rate["_stratum_2"] == 0.5
    assert size_str_wald.resp_rate["_stratum_3"] == 0.5
    assert size_str_wald.resp_rate["_stratum_4"] == 0.5
    assert size_str_wald.resp_rate["_stratum_5"] == 0.5


def test_size_str_wald_df():
    size_str_wald.calculate(target=0.80, half_ci=0.10, number_strata=5)
    size_df = size_str_wald.to_dataframe()
    assert size_df.shape[0] == 5
    assert (
        size_df.columns == ["_param", "_stratum", "_target", "_sigma", "_half_ci", "_samp_size"]
    ).all()


def test_size_str_wald_df_with_resp_rate():
    size_str_wald.calculate(target=0.80, half_ci=0.10, number_strata=5, resp_rate=0.56)
    size_df = size_str_wald.to_dataframe()
    assert size_df.shape[0] == 5
    assert (
        size_df.columns == ["_param", "_stratum", "_target", "_sigma", "_half_ci", "_samp_size"]
    ).all()


## Fleiss' method

size_nat_fleiss = SampleSize(method=SizeMethod.fleiss)


def test_size_nat_fleiss_basics():
    assert size_nat_fleiss.param == PopParam.prop
    assert size_nat_fleiss.method == SizeMethod.fleiss
    assert size_nat_fleiss.strat == False


def test_size_nat_fleiss_size1a():
    size_nat_fleiss.calculate(target=0.80, half_ci=0.10)
    assert size_nat_fleiss.samp_size == 88
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.80
    assert size_nat_fleiss.half_ci == 0.1
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size1a_with_resp_rate():
    size_nat_fleiss.calculate(target=0.80, half_ci=0.10, resp_rate=0.5)
    assert size_nat_fleiss.samp_size == 176
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.80
    assert size_nat_fleiss.half_ci == 0.1
    assert size_nat_fleiss.resp_rate == 0.5


def test_size_nat_fleiss_size1b():
    size_nat_fleiss.calculate(target=0.20, half_ci=0.10)
    assert size_nat_fleiss.samp_size == 88
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.20
    assert size_nat_fleiss.half_ci == 0.1
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size1b_with_resp_rate():
    size_nat_fleiss.calculate(target=0.20, half_ci=0.10, resp_rate=0.6)
    assert size_nat_fleiss.samp_size == 147
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.20
    assert size_nat_fleiss.half_ci == 0.1
    assert size_nat_fleiss.resp_rate == 0.6


def test_size_nat_fleiss_size2a():
    size_nat_fleiss.calculate(target=0.95, half_ci=0.06)
    assert size_nat_fleiss.samp_size == 132
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.95
    assert size_nat_fleiss.half_ci == 0.06
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size2b():
    size_nat_fleiss.calculate(target=0.05, half_ci=0.06)
    assert size_nat_fleiss.samp_size == 132
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.05
    assert size_nat_fleiss.half_ci == 0.06
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size3():
    size_nat_fleiss.calculate(target=0.70, half_ci=0.03)
    assert size_nat_fleiss.samp_size == 1097
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.70
    assert size_nat_fleiss.half_ci == 0.03
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size4():
    size_nat_fleiss.calculate(target=0.85, half_ci=0.03)
    assert size_nat_fleiss.samp_size == 663
    assert size_nat_fleiss.deff_c == 1.0
    assert size_nat_fleiss.target == 0.85
    assert size_nat_fleiss.half_ci == 0.03
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size_with_deff1a():
    size_nat_fleiss.calculate(target=0.80, half_ci=0.10, deff=1.5)
    assert size_nat_fleiss.samp_size == 132
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.80
    assert size_nat_fleiss.half_ci == 0.1
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size_with_deff1a_with_resp_rate():
    size_nat_fleiss.calculate(target=0.80, half_ci=0.10, deff=1.5, resp_rate=0.4)
    assert size_nat_fleiss.samp_size == 329
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.80
    assert size_nat_fleiss.half_ci == 0.1
    assert size_nat_fleiss.resp_rate == 0.4


def test_size_nat_fleiss_size_with_deff1b_with_resp_rate():
    size_nat_fleiss.calculate(target=0.20, half_ci=0.10, deff=1.5, resp_rate=0.4)
    assert size_nat_fleiss.samp_size == 329
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.20
    assert size_nat_fleiss.half_ci == 0.1
    assert size_nat_fleiss.resp_rate == 0.4


def test_size_nat_fleiss_size_with_deff2a():
    size_nat_fleiss.calculate(target=0.95, half_ci=0.06, deff=1.5)
    assert size_nat_fleiss.samp_size == 197
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.95
    assert size_nat_fleiss.half_ci == 0.06
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size_with_deff2b():
    size_nat_fleiss.calculate(target=0.05, half_ci=0.06, deff=1.5)
    assert size_nat_fleiss.samp_size == 197
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.05
    assert size_nat_fleiss.half_ci == 0.06
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size_with_deff3():
    size_nat_fleiss.calculate(target=0.70, half_ci=0.03, deff=1.5)
    assert size_nat_fleiss.samp_size == 1646
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.70
    assert size_nat_fleiss.half_ci == 0.03
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_size_with_deff4():
    size_nat_fleiss.calculate(target=0.85, half_ci=0.03, deff=1.5)
    assert size_nat_fleiss.samp_size == 994
    assert size_nat_fleiss.deff_c == 1.5
    assert size_nat_fleiss.target == 0.85
    assert size_nat_fleiss.half_ci == 0.03
    assert size_nat_fleiss.resp_rate == 1.0


def test_size_nat_fleiss_df1():
    size_nat_fleiss.calculate(target=0.80, half_ci=0.10)
    size_df = size_nat_fleiss.to_dataframe()
    assert (size_df.columns == ["_param", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_nat_fleiss_df1_with_resp_rate():
    size_nat_fleiss.calculate(target=0.80, half_ci=0.10, resp_rate=0.33)
    size_df = size_nat_fleiss.to_dataframe()
    assert (size_df.columns == ["_param", "_target", "_sigma", "_half_ci", "_samp_size"]).all()


def test_size_nat_fleiss_df2():
    size_nat_fleiss.calculate(target=0.80, half_ci=0.10)
    size_df = size_nat_fleiss.to_dataframe(["param", "prop", "sigma", "half_ci", "size"])
    assert (size_df.columns == ["param", "prop", "sigma", "half_ci", "size"]).all()


def test_size_nat_fleiss_df2_with_resp_rate():
    size_nat_fleiss.calculate(target=0.80, half_ci=0.10, resp_rate=0.44)
    size_df = size_nat_fleiss.to_dataframe(["param", "prop", "sigma", "half_ci", "size"])
    assert (size_df.columns == ["param", "prop", "sigma", "half_ci", "size"]).all()


## Fleiss' method - stratified
size_str_fleiss = SampleSize(param=PopParam.prop, method=SizeMethod.fleiss, strat=True)

target2 = {"stratum1": 0.95, "stratum2": 0.70, "stratum3": 0.30}
half_ci2 = {"stratum1": 0.03, "stratum2": 0.10, "stratum3": 0.05}
deff2 = {"stratum1": 1, "stratum2": 1.5, "stratum3": 2.5}
resp_rate2 = {"stratum1": 1, "stratum2": 0.5, "stratum3": 0.75}


def test_size_str_fleiss_basics():
    assert size_str_fleiss.param == PopParam.prop
    assert size_str_fleiss.method == SizeMethod.fleiss
    assert size_str_fleiss.strat == True


def test_size_str_fleiss_size1():
    size_str_fleiss.calculate(target=target2, half_ci=0.10)
    assert size_str_fleiss.samp_size["stratum1"] == 70
    assert size_str_fleiss.samp_size["stratum2"] == 103
    assert size_str_fleiss.samp_size["stratum3"] == 103
    assert size_str_fleiss.resp_rate["stratum1"] == 1
    assert size_str_fleiss.resp_rate["stratum2"] == 1
    assert size_str_fleiss.resp_rate["stratum3"] == 1


def test_size_str_fleiss_size1_with_resp_rate1():
    size_str_fleiss.calculate(target=target2, half_ci=0.10, resp_rate=0.5)
    assert size_str_fleiss.samp_size["stratum1"] == 140
    assert size_str_fleiss.samp_size["stratum2"] == 205
    assert size_str_fleiss.samp_size["stratum3"] == 205
    assert size_str_fleiss.resp_rate["stratum1"] == 0.5
    assert size_str_fleiss.resp_rate["stratum2"] == 0.5
    assert size_str_fleiss.resp_rate["stratum3"] == 0.5


def test_size_str_fleiss_size1_with_resp_rate2():
    size_str_fleiss.calculate(target=target2, half_ci=0.10, resp_rate=resp_rate2)
    assert size_str_fleiss.samp_size["stratum1"] == 70
    assert size_str_fleiss.samp_size["stratum2"] == 205
    assert size_str_fleiss.samp_size["stratum3"] == 137
    assert size_str_fleiss.resp_rate["stratum1"] == 1
    assert size_str_fleiss.resp_rate["stratum2"] == 0.5
    assert size_str_fleiss.resp_rate["stratum3"] == 0.75


def test_size_str_fleiss_size2():
    size_str_fleiss.calculate(target=0.8, half_ci=half_ci2)
    assert size_str_fleiss.samp_size["stratum1"] == 788
    assert size_str_fleiss.samp_size["stratum2"] == 88
    assert size_str_fleiss.samp_size["stratum3"] == 306


def test_size_str_fleiss_size2_with_resp_rate1():
    size_str_fleiss.calculate(target=0.8, half_ci=half_ci2, resp_rate=0.7)
    assert size_str_fleiss.samp_size["stratum1"] == 1125
    assert size_str_fleiss.samp_size["stratum2"] == 126
    assert size_str_fleiss.samp_size["stratum3"] == 437
    assert size_str_fleiss.resp_rate["stratum1"] == 0.7
    assert size_str_fleiss.resp_rate["stratum2"] == 0.7
    assert size_str_fleiss.resp_rate["stratum3"] == 0.7


def test_size_str_fleiss_size2_with_resp_rate2():
    size_str_fleiss.calculate(target=0.8, half_ci=half_ci2, resp_rate=resp_rate2)
    assert size_str_fleiss.samp_size["stratum1"] == 788
    assert size_str_fleiss.samp_size["stratum2"] == 176
    assert size_str_fleiss.samp_size["stratum3"] == 408
    assert size_str_fleiss.resp_rate["stratum1"] == 1
    assert size_str_fleiss.resp_rate["stratum2"] == 0.5
    assert size_str_fleiss.resp_rate["stratum3"] == 0.75


def test_size_str_fleiss_size3():
    size_str_fleiss.calculate(target=0.8, half_ci=0.10, deff=deff2)
    assert size_str_fleiss.samp_size["stratum1"] == 88
    assert size_str_fleiss.samp_size["stratum2"] == 132
    assert size_str_fleiss.samp_size["stratum3"] == 220


def test_size_str_fleiss_size4():
    size_str_fleiss.calculate(target=target2, half_ci=half_ci2, deff=deff2)
    assert size_str_fleiss.samp_size["stratum1"] == 354
    assert size_str_fleiss.samp_size["stratum2"] == 154
    assert size_str_fleiss.samp_size["stratum3"] == 1002


def test_size_str_fleiss_size5():
    size_str_fleiss.calculate(target=0.8, half_ci=0.1, deff=1.5, number_strata=5)
    assert size_str_fleiss.samp_size["_stratum_1"] == 132
    assert size_str_fleiss.samp_size["_stratum_2"] == 132
    assert size_str_fleiss.samp_size["_stratum_3"] == 132
    assert size_str_fleiss.samp_size["_stratum_4"] == 132
    assert size_str_fleiss.samp_size["_stratum_5"] == 132


def test_size_str_fleiss_df1():
    size_str_fleiss.calculate(target=0.80, half_ci=0.10, number_strata=5)
    size_df = size_str_fleiss.to_dataframe()
    assert size_df.shape[0] == 5
    assert (
        size_df.columns == ["_param", "_stratum", "_target", "_sigma", "_half_ci", "_samp_size"]
    ).all()


def test_size_str_fleiss_df1_with_resp_rate():
    size_str_fleiss.calculate(target=0.80, half_ci=0.10, number_strata=5, resp_rate=0.4)
    size_df = size_str_fleiss.to_dataframe()
    assert size_df.shape[0] == 5
    assert (
        size_df.columns == ["_param", "_stratum", "_target", "_sigma", "_half_ci", "_samp_size"]
    ).all()


def test_size_str_fleiss_df2():
    size_str_fleiss.calculate(target=0.80, half_ci=0.10, number_strata=5)
    size_df = size_str_fleiss.to_dataframe(["param", "str", "prop", "sigma", "half_ci", "size"])
    assert size_df.shape[0] == 5
    assert (size_df.columns == ["param", "str", "prop", "sigma", "half_ci", "size"]).all()


def test_size_str_fleiss_df2_with_resp_rate():
    size_str_fleiss.calculate(target=0.80, half_ci=0.10, number_strata=5, resp_rate=0.6)
    size_df = size_str_fleiss.to_dataframe(["param", "str", "prop", "sigma", "half_ci", "size"])
    assert size_df.shape[0] == 5
    assert (size_df.columns == ["param", "str", "prop", "sigma", "half_ci", "size"]).all()
