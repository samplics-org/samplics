import pytest

import pandas as pd

from samplics.categorical.comparison import Ttest

auto = pd.read_csv("./tests/categorical/auto.csv")

y = auto["mpg"]
make = auto["make"]
foreign = auto["make"]


@pytest.mark.xfail(strict=True, reason="Parameters 'known_mean' or 'group' must be provided!")
def test_one_sample_wrong_specifications1():
    one_sample_wrong = Ttest()
    one_sample_wrong.compare(y)


@pytest.mark.xfail(strict=True, reason="Parameters 'known_mean' or 'group' must be provided!")
def test_one_sample_wrong_specifications2():
    one_sample_wrong = Ttest("one-sample")
    one_sample_wrong.compare(y)


@pytest.mark.xfail(
    strict=True,
    reason="Parameter 'type' must be equal to 'one-sample', 'two-sample' or 'many-sample'!",
)
def test_one_sample_wrong_specifications3():
    one_sample_wrong = Ttest("two-sample")
    one_sample_wrong.compare(y, known_mean=0, group=make)


one_sample_auto = Ttest(type="one-sample")


def test_one_sample_wrong_specifications2():
    one_sample_auto.compare(y, known_mean=20)


# breakpoint()