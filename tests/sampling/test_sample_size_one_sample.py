import numpy as np

from samplics.sampling import (
    SampleSizeMeanOneSample,
    SampleSizePropOneSample,
    calculate_power,
)


def test_calculate_power_two_sides():
    assert np.isclose(
        calculate_power(two_sides=True, delta=2, sigma=3, samp_size=18, alpha=0.05),
        0.807,
        0.001,
    )
    assert np.isclose(
        calculate_power(two_sides=True, delta=0.8, sigma=3, samp_size=111, alpha=0.05),
        0.802,
        0.001,
    )
    assert np.isclose(
        calculate_power(two_sides=True, delta=0.6, sigma=3, samp_size=197, alpha=0.05),
        0.801,
        0.001,
    )
    assert np.isclose(
        calculate_power(two_sides=True, delta=0.2, sigma=3, samp_size=1766, alpha=0.05),
        0.800,
        0.001,
    )


def test_calculate_power_one_side():
    assert np.isclose(
        calculate_power(two_sides=False, delta=2, sigma=3, samp_size=14, alpha=0.05),
        0.802,
        0.001,
    )
    assert np.isclose(
        calculate_power(two_sides=False, delta=0.8, sigma=3, samp_size=87, alpha=0.05),
        0.800,
        0.001,
    )
    assert np.isclose(
        calculate_power(two_sides=False, delta=0.6, sigma=3, samp_size=155, alpha=0.05),
        0.801,
        0.001,
    )
    assert np.isclose(
        calculate_power(
            two_sides=False, delta=0.2, sigma=3, samp_size=1392, alpha=0.05
        ),
        0.801,
        0.001,
    )


size_diff_mean_one_side = SampleSizeMeanOneSample(two_sides=False)


def test_ss_diff_mean_one_side1():
    size_diff_mean_one_side.calculate(
        mean_0=50, mean_1=52, sigma=3, alpha=0.05, power=0.80
    )
    assert size_diff_mean_one_side.samp_size == 14
    # assert np.isclose(size_diff_mean_one_side.actual_power, 0.802, atol=0.001)


def test_ss_diff_mean_one_side1_with_resp_rate():
    size_diff_mean_one_side.calculate(
        mean_0=50, mean_1=52, sigma=3, alpha=0.05, power=0.80, resp_rate=0.5
    )
    assert size_diff_mean_one_side.samp_size == 28
    assert size_diff_mean_one_side.resp_rate == 0.5
    # assert np.isclose(size_diff_mean_one_side.actual_power, 0.802, atol=0.001)


def test_ss_diff_mean_one_side2():
    size_diff_mean_one_side.calculate(
        mean_0=50, mean_1=50.8, sigma=3, alpha=0.05, power=0.80
    )
    assert size_diff_mean_one_side.samp_size == 87
    assert np.isclose(size_diff_mean_one_side.actual_power, 0.800, atol=0.001)


def test_ss_diff_mean_one_side2_with_resp_rate():
    size_diff_mean_one_side.calculate(
        mean_0=50, mean_1=50.8, sigma=3, alpha=0.05, power=0.80, resp_rate=0.75
    )
    assert size_diff_mean_one_side.samp_size == 116
    assert size_diff_mean_one_side.resp_rate == 0.75
    # assert np.isclose(size_diff_mean_one_side.actual_power, 0.800, atol=0.001)


def test_ss_diff_mean_one_side3():
    size_diff_mean_one_side.calculate(
        mean_0=50, mean_1=50.6, sigma=3, alpha=0.05, power=0.80
    )
    assert size_diff_mean_one_side.samp_size == 155
    assert np.isclose(size_diff_mean_one_side.actual_power, 0.801, atol=0.001)


def test_ss_diff_mean_one_side4():
    size_diff_mean_one_side.calculate(
        mean_0=50, mean_1=50.2, sigma=3, alpha=0.05, power=0.80
    )
    assert size_diff_mean_one_side.samp_size == 1392
    assert np.isclose(size_diff_mean_one_side.actual_power, 0.801, atol=0.001)


size_diff_mean_two_sides = SampleSizeMeanOneSample(two_sides=True)


def test_ss_diff_mean_two_sides1():
    size_diff_mean_two_sides.calculate(
        mean_0=50, mean_1=52, sigma=3, alpha=0.05, power=0.80
    )
    assert size_diff_mean_two_sides.samp_size == 18
    assert np.isclose(size_diff_mean_two_sides.actual_power, 0.807, atol=0.001)


def test_ss_diff_mean_two_sides2():
    size_diff_mean_two_sides.calculate(
        mean_0=50, mean_1=50.8, sigma=3, alpha=0.05, power=0.80
    )
    assert size_diff_mean_two_sides.samp_size == 111
    assert np.isclose(size_diff_mean_two_sides.actual_power, 0.802, atol=0.001)


def test_ss_diff_mean_two_sides3():
    size_diff_mean_two_sides.calculate(
        mean_0=50, mean_1=50.6, sigma=3, alpha=0.05, power=0.80
    )
    assert size_diff_mean_two_sides.samp_size == 197
    assert np.isclose(size_diff_mean_two_sides.actual_power, 0.802, atol=0.001)


def test_ss_diff_mean_two_sides4():
    size_diff_mean_two_sides.calculate(
        mean_0=50, mean_1=50.2, sigma=3, alpha=0.05, power=0.80
    )
    assert size_diff_mean_two_sides.samp_size == 1766
    assert np.isclose(size_diff_mean_two_sides.actual_power, 0.800, atol=0.001)


size_diff_prop_one_side = SampleSizePropOneSample(two_sides=False)


def test_ss_diff_prop_one_side1():
    size_diff_prop_one_side.calculate(
        prop_0=0.3, prop_1=0.5, delta=-0.10, alpha=0.05, power=0.80
    )
    assert size_diff_prop_one_side.samp_size == 18
    assert np.isclose(size_diff_prop_one_side.actual_power, 0.5208, atol=0.001)


# def test_ss_diff_prop_one_side1():
#     size_diff_prop_one_side.calculate(
#         prop_0=0.2, prop_1=0.1, arcsin=True, continuity=False, alpha=0.05, power=0.90
#     )
#     assert size_diff_prop_one_side.samp_size == 107
#     assert np.isclose(size_diff_prop_one_side.actual_power, 0.802, atol=0.001)

size_diff_prop_two_sides = SampleSizePropOneSample()


def test_ss_diff_prop_two_sides1():
    size_diff_prop_two_sides.calculate(prop_0=0.3, prop_1=0.5, alpha=0.05, power=0.80)
    assert size_diff_prop_two_sides.samp_size == 50
    assert np.isclose(size_diff_prop_two_sides.actual_power, 0.8074, atol=0.001)


def test_ss_diff_prop_two_sides2():
    size_diff_prop_two_sides.calculate(
        prop_0=0.6, prop_1=0.6, delta=-0.20, alpha=0.05, power=0.80
    )
    assert size_diff_prop_two_sides.samp_size == 52
    assert np.isclose(size_diff_prop_two_sides.actual_power, 0.0500, atol=0.001)
