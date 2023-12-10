import numpy as np
import pytest

from samplics.utils.hadamard import hadamard


def test_hadarmard_2():
    had_mat2 = hadamard(2)
    test2 = had_mat2 == np.array([[1, 1], [1, -1]])
    assert test2.all()


@pytest.mark.parametrize("n", [12, 20, 24, 28])
def test_hadarmard(n):
    had_mat = hadamard(n)
    prod = had_mat @ had_mat.T
    assert (np.diag(prod) == np.repeat(n, n)).all()
    assert np.isclose(abs(np.linalg.det(had_mat)), np.power(n, n / 2), atol=1e-6)
