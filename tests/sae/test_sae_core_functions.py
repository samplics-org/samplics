import numpy as np

from samplics.sae.sae_core_functions import (
    covariance,
    fixed_coefficients,
    inverse_covariance,
    iterative_fisher_scoring,
    log_det_covariance,
    log_likelihood,
    partial_derivatives,
)

np.random.seed(12345)

# model parameters
scale = 1
sigma2e = 1**2
sigma2u = 0.25**2

# Population sizes
N = 1000
nb_areas = 10

error = np.random.normal(loc=0, scale=(scale**2) * (sigma2e**0.5), size=N)
area = np.sort(np.random.choice(range(1, nb_areas + 1), N))
areas, Nd = np.unique(area, return_counts=True)

random_effects = np.random.normal(loc=0, scale=sigma2u ** (1 / 2), size=nb_areas)
total_error = np.repeat(random_effects, Nd) + error

# Auxiliary information
p1 = 0.3 + areas / np.max(areas)
p1 = p1 / sum(p1)
p1 = p1 / (1.2 * max(p1))

X1 = np.array([])
for k, s in enumerate(Nd):
    Xk = np.random.binomial(1, p=p1[k], size=s)
    X1 = np.append(X1, Xk)
X2 = 0.01 + np.random.beta(0.5, 1, N)
X3 = np.random.binomial(1, p=0.6, size=N)
X = np.column_stack((np.ones(N), X1, X2, X3))


beta = np.array([1, 3, -3, 3])
y = X @ beta + total_error


scale = np.ones(y.size)
est_beta = fixed_coefficients(y, X, area, 1, 0.25, scale)
est_covariance = covariance(area, 1, 0.25, scale)
est_inv_covariance = inverse_covariance(area, 1, 0.25, scale)
est_log_det_covariance = log_det_covariance(area, 1, 0.25, scale)


def test_fixed_coefficients():
    assert np.isclose(est_beta, [0.9759, 2.9918, -2.9886, 3.0172], atol=0.3).all()


def test_covariance():
    assert est_covariance.shape == (N, N)
    assert est_covariance[0, 0] == 1.0625
    assert est_covariance[N - 1, N - 1] == 1.0625
    assert est_covariance[0, 1] == 0.0625
    assert est_covariance[1, N - 1] == 0


def test_inv_covariance():
    assert est_inv_covariance.shape == (N, N)
    assert np.isclose(est_inv_covariance[0, 0], 0.9892473, atol=1e-3)
    assert np.isclose(est_inv_covariance[N - 1, N - 1], 0.991667, atol=1e-3)
    assert np.isclose(est_inv_covariance[0, 1], -0.010753, atol=1e-3)
    assert np.isclose(est_inv_covariance[1, N - 1], 0.0, atol=1e-3)


def test_log_det_covariance():
    assert np.isclose(est_log_det_covariance, 19.76636, atol=1e-3)


## REML method
est_log_likelihod = log_likelihood(
    "REML", y, X, beta, est_inv_covariance, est_log_det_covariance
)
est_partial_deriv, est_info_matrix = partial_derivatives(
    "REML", area, y, X, 1, 0.25, scale
)


def test_log_likehood():
    assert np.isclose(est_log_likelihod, -1450.84762, atol=1e-3)


def test_partial_derivatives():
    assert np.isclose(est_partial_deriv[0], -16.2354493, atol=1e-3)
    assert np.isclose(est_partial_deriv[1], 5.22990259e-13, atol=1e-3)


def test_info_matrix():
    assert np.isclose(est_info_matrix[0, 0], 493.5917602, atol=1e-3)
    assert np.isclose(est_info_matrix[1, 1], 5.30054753e-25, atol=1e-3)
    assert np.isclose(est_info_matrix[1, 0], -2.84637279e-19, atol=1e-3)
    assert np.isclose(est_info_matrix[0, 1], -2.84637279e-19, atol=1e-3)


def test_iterative_fisher():
    sigma2, inv_info_mat, iterations, tolerance, convergence = iterative_fisher_scoring(
        "REML", area, y, X, 1, 0.25, scale, 0.01, 0.01, 5
    )


## REML method
est_log_likelihod_ml = log_likelihood(
    "ML", y, X, beta, est_inv_covariance, est_log_det_covariance
)
est_partial_deriv_ml, est_info_matrix_ml = partial_derivatives(
    "ML", area, y, X, 1, 0.25, scale
)


def test_log_likehood2():
    assert np.isclose(est_log_likelihod_ml, -1410.6938996, atol=1e-3)


def test_partial_derivatives2():
    assert np.isclose(est_partial_deriv_ml[0], -17.8333010, atol=1e-3)
    assert np.isclose(est_partial_deriv_ml[1], -9.58866729, atol=1e-3)


def test_info_matrix2():
    assert np.isclose(est_info_matrix_ml[0, 0], 495.09776441, atol=1e-3)
    assert np.isclose(est_info_matrix_ml[1, 1], 948.759416, atol=1e-3)
    assert np.isclose(est_info_matrix_ml[1, 0], 9.5691529, atol=1e-3)
    assert np.isclose(est_info_matrix_ml[0, 1], 9.56915294, atol=1e-3)


def test_iterative_fisher2():
    sigma2, inv_info_mat, iterations, tolerance, convergence = iterative_fisher_scoring(
        "ML", area, y, X, 1, 0.25, scale, 0.01, 0.01, 5
    )
