from samplics.sampling import SampleSizeMeanTwoSample, SampleSizePropTwoSample


size_diff_mean_one_side = SampleSizeMeanTwoSample(two_sides=False)


def test_ss_diff_mean_one_side1():
    size_diff_mean_one_side.calculate(
        mean_1=0.50, mean_2=0.50, sigma_1=0.10, delta=-0.05, alpha=0.05, power=0.80
    )
    assert size_diff_mean_one_side.samp_size == (50, 50)
    # assert np.isclose(size_diff_mean_one_side.actual_power, 0.802, atol=0.001)


size_diff_mean_two_sides = SampleSizeMeanTwoSample(two_sides=True)


def test_ss_diff_mean_two_side1():
    size_diff_mean_two_sides.calculate(
        mean_1=0.50, mean_2=0.55, sigma_1=0.10, alpha=0.05, power=0.80
    )
    assert size_diff_mean_two_sides.samp_size == (63, 63)
    # assert np.isclose(size_diff_mean_two_sides.actual_power, 0.802, atol=0.001)


def test_ss_diff_mean_two_side2():
    size_diff_mean_two_sides.calculate(
        mean_1=0.50, mean_2=0.51, sigma_1=0.10, delta=0.05, alpha=0.05, power=0.80
    )
    assert size_diff_mean_two_sides.samp_size == (108, 108)
    # assert np.isclose(size_diff_mean_two_sides.actual_power, 0.802, atol=0.001)


size_diff_prop_one_side = SampleSizePropTwoSample(two_sides=False)


def test_ss_diff_prop_one_side1():
    size_diff_prop_one_side.calculate(
        prop_1=0.65, prop_2=0.85, delta=-0.10, alpha=0.05, power=0.80
    )
    assert size_diff_prop_one_side.samp_size == (25, 25)
    # assert np.isclose(size_diff_prop_one_side.actual_power, 0.802, atol=0.001)


def test_ss_diff_prop_one_side2():
    size_diff_prop_one_side.calculate(
        prop_1=0.65, prop_2=0.85, delta=0.05, alpha=0.05, power=0.80
    )
    assert size_diff_prop_one_side.samp_size == (98, 98)
    # assert np.isclose(size_diff_prop_one_side.actual_power, 0.802, atol=0.001)


size_diff_prop_two_side = SampleSizePropTwoSample(two_sides=True)


def test_ss_diff_prop_two_side1():
    size_diff_prop_two_side.calculate(prop_1=0.65, prop_2=0.85, alpha=0.05, power=0.80)
    assert size_diff_prop_two_side.samp_size == (70, 70)
    # assert np.isclose(size_diff_prop_two_side.actual_power, 0.802, atol=0.001)


def test_ss_diff_prop_two_side2():
    size_diff_prop_two_side.calculate(
        prop_1=0.75, prop_2=0.80, delta=0.20, alpha=0.05, power=0.80
    )
    assert size_diff_prop_two_side.samp_size == (133, 133)
    # assert np.isclose(size_diff_prop_two_side.actual_power, 0.802, atol=0.001)
