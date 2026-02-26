from typing import Any

import numpy as np


class Estimator:
    """
    Represents an estimator of a mean with variance propagation support.
    """

    def __init__(
        self,
        mean: float,
        variance: float,
        n: int = 1,
    ):
        self.mean = float(mean)
        self.variance = float(max(variance, 0.0))
        self.n = int(max(n, 1))

    @property
    def std(self):
        return np.sqrt(self.variance)

    @classmethod
    def from_samples(
        cls, samples: np.ndarray[Any, np.dtype[Any]], mode: str = "empirical"
    ):
        n = len(samples)
        if n == 0:
            return cls(0.0, 0.0, 1)

        mean = np.mean(samples)

        if mode == "poisson":
            variance = float(mean / n)
        else:
            std = np.std(samples, ddof=1)
            variance = float((std**2) / n)

        return cls(float(mean), variance, n)

    @classmethod
    def from_mean_std(cls, mean: float, std: float, n: int, mode: str = "empirical"):
        if n <= 0:
            return cls(mean, 0.0, 1)

        if mode == "poisson":
            variance = mean / n
        else:
            variance = (std**2) / n

        return cls(mean, variance, n)

    def ratio(self, other, cov: float = 0.0, eps: float = 1e-12):
        """
        First-order propagation of ratio self / other.
        """
        if abs(other.mean) < eps:
            return Estimator(0.0, 0.0, 1)

        mean = self.mean / other.mean

        grad1 = 1.0 / other.mean
        grad2 = -self.mean / (other.mean**2)

        var = (
            grad1**2 * self.variance
            + grad2**2 * other.variance
            + 2.0 * grad1 * grad2 * cov
        )

        return Estimator(mean, var, 1)

    def scale(self, factor: float):
        return Estimator(
            self.mean * factor,
            self.variance * (factor**2),
            self.n,
        )

    def subtract_constant(self, c: float):
        return Estimator(
            self.mean - c,
            self.variance,
            self.n,
        )


class MultiEstimator:
    """
    Multivariate estimator supporting full covariance propagation.
    """

    def __init__(
        self, means: np.ndarray[Any, np.dtype[Any]], cov: np.ndarray[Any, np.dtype[Any]]
    ):
        self.means = np.atleast_1d(means).astype(float)
        self.cov = np.atleast_2d(cov).astype(float)

        if self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError("Covariance matrix must be square")

        if self.cov.shape[0] != self.means.shape[0]:
            raise ValueError("Covariance dimension mismatch")

    # -------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------

    @classmethod
    def from_independent(cls, estimators):
        """
        Combine independent scalar Estimators into block-diagonal covariance.
        """
        means = np.array([e.mean for e in estimators])
        cov = np.diag([e.variance for e in estimators])
        return cls(means, cov)

    @classmethod
    def from_samples(cls, samples_matrix):
        """
        samples_matrix shape: (n_samples, n_variables)
        """
        means = np.mean(samples_matrix, axis=0)
        cov = np.cov(samples_matrix, rowvar=False) / samples_matrix.shape[0]
        return cls(means, cov)

    # -------------------------------------------------------
    # General propagation
    # -------------------------------------------------------

    def propagate(self, func, jacobian_func):
        """
        Propagate through arbitrary function using Jacobian.
        """

        mean_out = func(self.means)
        J = jacobian_func(self.means)  # shape (m, n)

        cov_out = J @ self.cov @ J.T

        return MultiEstimator(mean_out, cov_out)

    @property
    def std(self):
        return np.sqrt(np.diag(self.cov))


def ratio_multivariate(est: MultiEstimator):
    """
    Assumes 2D estimator: [X, Y]
    Returns estimator of X/Y
    """

    def func(m):
        return np.array([m[0] / m[1]])

    def jac(m):
        X, Y = m
        return np.array([[1.0 / Y, -X / (Y**2)]])

    return est.propagate(func, jac)
