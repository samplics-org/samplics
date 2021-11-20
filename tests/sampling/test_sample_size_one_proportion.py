import numpy as np

from samplics.sampling import SampleSizeOneProportion, power_for_proportion


size_diff_prop_one_side = SampleSizeOneProportion(two_sides=False)


def test_ss_diff_prop_one_side1():
    size_diff_prop_one_side.calculate(
        prop_0=0.2, prop_1=0.1, arcsin=True, continuity=False, alpha=0.05, power=0.90
    )
    assert size_diff_prop_one_side.samp_size == 107
    # assert np.isclose(size_diff_prop_one_side.actual_power, 0.802, atol=0.001)


# def test_ss_diff_mean_one_side2():
#     size_diff_mean_one_side.calculate(mean_0=50, mean_1=50.8, sigma=3, alpha=0.05, power=0.80)
#     assert size_diff_mean_one_side.samp_size == 87
#     assert np.isclose(size_diff_mean_one_side.actual_power, 0.800, atol=0.001)


# def test_ss_diff_mean_one_side3():
#     size_diff_mean_one_side.calculate(mean_0=50, mean_1=50.6, sigma=3, alpha=0.05, power=0.80)
#     assert size_diff_mean_one_side.samp_size == 155
#     assert np.isclose(size_diff_mean_one_side.actual_power, 0.801, atol=0.001)


# def test_ss_diff_mean_one_side4():
#     size_diff_mean_one_side.calculate(mean_0=50, mean_1=50.2, sigma=3, alpha=0.05, power=0.80)
#     assert size_diff_mean_one_side.samp_size == 1392
#     assert np.isclose(size_diff_mean_one_side.actual_power, 0.801, atol=0.001)


size_diff_mean_two_sides = SampleSizeOneProportion()


def test_ss_diff_mean_two_sides1():
    size_diff_mean_two_sides.calculate(
        prop_0=0.2, prop_1=0.1, arcsin=True, continuity=False, alpha=0.05, power=0.90
    )
    assert size_diff_mean_two_sides.samp_size == 522
    assert np.isclose(size_diff_mean_two_sides.actual_power, 0.9613, atol=0.001)


# def test_ss_diff_mean_two_sides2():
#     size_diff_mean_two_sides.calculate(mean_0=50, mean_1=50.8, sigma=3, alpha=0.05, power=0.80)
#     assert size_diff_mean_two_sides.samp_size == 111
#     assert np.isclose(size_diff_mean_two_sides.actual_power, 0.802, atol=0.001)


# def test_ss_diff_mean_two_sides3():
#     size_diff_mean_two_sides.calculate(mean_0=50, mean_1=50.6, sigma=3, alpha=0.05, power=0.80)
#     assert size_diff_mean_two_sides.samp_size == 197
#     assert np.isclose(size_diff_mean_two_sides.actual_power, 0.802, atol=0.001)


# def test_ss_diff_mean_two_sides4():
#     size_diff_mean_two_sides.calculate(mean_0=50, mean_1=50.2, sigma=3, alpha=0.05, power=0.80)
#     assert size_diff_mean_two_sides.samp_size == 1766
#     assert np.isclose(size_diff_mean_two_sides.actual_power, 0.800, atol=0.001)
