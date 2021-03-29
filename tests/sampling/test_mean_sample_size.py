import pytest

from samplics.sampling import OneMeanSampleSize


region = ["Dakar", "Kaolack", "Ziguinchor"]
pop_size = {"Dakar": 500, "Kaolack": 300, "Ziguinchor": 200}

reference_mean = {"Dakar": 50, "Kaolack": 65, "Ziguinchor": 30}
targeted_mean = {"Dakar": 52, "Kaolack": 60, "Ziguinchor": 33}
stddev = {"Dakar": 3, "Kaolack": 4, "Ziguinchor": 5}


## Some failing scenarios


@pytest.mark.xfail(reason="Stratified designs must have dictionaries or number_strata")
def test_one_mean_sample_size_stratified_fail1():
    one_mean_str = OneMeanSampleSize(stratification=True)
    one_mean_str.calculate(targeted_mean=52, reference_mean=50, stddev=3)


@pytest.mark.xfail(reason="Not stratified designs must not have dictionaries")
def test_one_mean_sample_size_stratified_fail2():
    one_mean_str = OneMeanSampleSize(stratification=False)
    one_mean_str.calculate(targeted_mean=targeted_mean, reference_mean=50, stddev=3)


## One sample size with one side - Not stratified
one_mean_not_str_oneside = OneMeanSampleSize(
    stratification=False, two_side=False, estimated_mean=True
)


def test_one_mean_sample_size_not_stratified_oneside_1():
    one_mean_not_str_oneside.calculate(targeted_mean=52, reference_mean=50, stddev=3)
    assert one_mean_not_str_oneside.samp_size == 16
    assert one_mean_not_str_oneside.power == pytest.approx(0.802239, 0.00001)


def test_one_mean_sample_size_not_stratified_oneside_2():
    one_mean_not_str_oneside.calculate(targeted_mean=52, reference_mean=50, stddev=3, beta=0.05)
    assert one_mean_not_str_oneside.samp_size == 25
    assert one_mean_not_str_oneside.power == pytest.approx(0.954341, 0.00001)


def test_one_mean_sample_size_not_stratified_oneside_3():
    one_mean_not_str_oneside.calculate(targeted_mean=52, reference_mean=50, stddev=3, beta=0.01)
    assert one_mean_not_str_oneside.samp_size == 36
    assert one_mean_not_str_oneside.power == pytest.approx(0.990742, 0.00001)


## One sample size with two side - Not stratified
one_mean_not_str_twoside = OneMeanSampleSize(
    stratification=False, two_side=True, estimated_mean=False
)


def test_one_mean_sample_size_not_stratified_twoside_1():
    one_mean_not_str_twoside.calculate(targeted_mean=52, reference_mean=50, stddev=3)
    # breakpoint()
    assert one_mean_not_str_twoside.samp_size == 18
    assert one_mean_not_str_twoside.power == pytest.approx(0.807430, 0.00001)


## One sample size with one side - Stratified
one_mean_str_oneside = OneMeanSampleSize(stratification=True, two_side=False, estimated_mean=False)


def test_one_mean_sample_size_stratified_oneside_1():
    one_mean_str_oneside.calculate(
        targeted_mean={"one": 52, "two": 52}, reference_mean=50, stddev=3
    )
    # breakpoint()
    assert one_mean_str_oneside.samp_size["one"] == 14
    assert one_mean_str_oneside.samp_size["two"] == 14
    assert one_mean_str_oneside.power["one"] == pytest.approx(0.802239, 0.00001)
    assert one_mean_str_oneside.power["two"] == pytest.approx(0.802239, 0.00001)


## One sample size with two side - Stratified
one_mean_str_twoside = OneMeanSampleSize(stratification=True, two_side=True, estimated_mean=False)


def test_one_mean_sample_size_stratified_twoside_1():
    one_mean_str_twoside.calculate(
        targeted_mean={"one": 52, "two": 52}, reference_mean=50, stddev=3
    )
    # breakpoint()
    assert one_mean_str_twoside.samp_size["one"] == 18
    assert one_mean_str_twoside.samp_size["two"] == 18
    assert one_mean_str_twoside.power["one"] == pytest.approx(0.807430, 0.00001)
    assert one_mean_str_twoside.power["two"] == pytest.approx(0.807430, 0.00001)


def test_one_mean_sample_size_stratified_twoside_2():
    one_mean_str_twoside.calculate(
        targeted_mean={"one": 52, "two": 50.6, "three": 50.2},
        reference_mean={"one": 50, "two": 50, "three": 50},
        stddev=3,
    )
    # breakpoint()
    assert one_mean_str_twoside.samp_size["one"] == 18
    assert one_mean_str_twoside.samp_size["two"] == 197
    assert one_mean_str_twoside.samp_size["three"] == 1766
    assert one_mean_str_twoside.power["one"] == pytest.approx(0.807430, 0.00001)
    assert one_mean_str_twoside.power["two"] == pytest.approx(0.801551, 0.00001)
    assert one_mean_str_twoside.power["three"] == pytest.approx(0.800001, 0.00001)


##
# def test_one_mean_sample_size_stratified_1():
