from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):
    def __init__(self, length_scale=1.0, amplitude=1.0, **params):
        self._length_scale = 1.0
        self._amplitude = 1.0
        self.set_params(length_scale=length_scale, amplitude=amplitude, **params)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def get_params(self):
        return {
            "length_scale": self._length_scale,
            "amplitude": self._amplitude,
            **{k: v for k, v in self.__dict__.items() if not k.startswith("_")},
        }

    def set_params(self, **params):
        for param, value in params.items():
            if param == "length_scale":
                self._set_length_scale(value)
            elif param == "amplitude":
                self._set_amplitude(value)
            else:
                setattr(self, param, value)

    def _set_length_scale(self, value):
        if value <= 0:
            raise ValueError("length_scale must be greater than 0")
        self._length_scale = value

    def _set_amplitude(self, value):
        if value == 0:
            raise ValueError("amplitude cannot be 0")
        self._amplitude = value

    @property
    def length_scale(self):
        return self._length_scale

    @length_scale.setter
    def length_scale(self, value):
        self._set_length_scale(value)

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._set_amplitude(value)

    def euclid_dist(self, x1, x2):
        return np.sqrt(np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T))

    @abstractmethod
    def compute(self, x1, x2):
        raise NotImplementedError("Subclasses must implement this method")


class RBFKernel(Kernel):
    def compute(self, x1, x2):
        return self.amplitude**2 * np.exp(-0.5 / self.length_scale**2 * self.euclid_dist(x1, x2) ** 2)


class PeriodicKernel(Kernel):
    def __init__(self, length_scale=1.0, amplitude=1.0, p=1.0):
        super().__init__(length_scale, amplitude, p=p)

    def compute(self, x1, x2):
        if self.p <= 0:
            raise ValueError("p must be greater than 0")
        return self.amplitude**2 * np.exp(-2 * np.sin(np.pi * self.euclid_dist(x1, x2) / self.p) / self.length_scale**2)


class RationalQuadraticKernel(Kernel):
    def __init__(self, length_scale=1.0, amplitude=1.0, alpha=1.0):
        super().__init__(length_scale, amplitude, alpha=alpha)

    def compute(self, x1, x2):
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")

        return self.amplitude**2 * (1 + self.euclid_dist(x1, x2) ** 2 / (2 * self.alpha * self.length_scale**2)) ** (
            -self.alpha
        )


class GaussianProcess:
    def __init__(self, X, Y, Y_err, noise=1e-3, kernel=None):
        self.X = X
        self.Y = Y
        self.Y_err = Y_err
        self.noise = noise
        self.kernel = kernel

    def __repr__(self):
        return f"{self.__class__.__name__}(X={self.X}, Y={self.Y}, noise={self.noise}, kernel={self.kernel})"

    def predict(self, X_pred):
        K = self.kernel.compute(self.X, self.X)
        K += np.diag(self.Y_err**2)
        K_inv = np.linalg.inv(K + self.noise * np.eye(len(self.X)))
        K_star = self.kernel.compute(self.X, X_pred)
        K_star_star = self.kernel.compute(X_pred, X_pred)

        mu = K_star.T.dot(K_inv).dot(self.Y)
        cov = K_star_star - K_star.T.dot(K_inv).dot(K_star)

        return mu.flatten(), np.diag(cov)

    def update_data(self, X_new, Y_new):
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))