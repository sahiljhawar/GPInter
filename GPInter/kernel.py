from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):
    def __init__(self, length_scale=1.0, amplitude=1.0, **params):
        self._length_scale = 1.0
        self._amplitude = 1.0
        self.set_params(length_scale=length_scale, amplitude=amplitude, **params)

    def __str__(self):
        params = [f"{key}={value}" for key, value in self.get_params().items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def __mul__(self, other):
        if isinstance(other, Kernel):
            return ProductKernel([self, other])
        else:
            raise ValueError("Can only multiply Kernel with another Kernel")

    def __add__(self, other):
        if isinstance(other, Kernel):
            return SumKernel([self, other])
        else:
            raise ValueError("Can only add Kernel with another Kernel")

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


class ProductKernel(Kernel):
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels

    def __str__(self):
        return " * ".join(str(kernel) for kernel in self.kernels)

    def __mul__(self, other):
        if isinstance(other, Kernel):
            if isinstance(other, ProductKernel):
                return ProductKernel(self.kernels + other.kernels)
            else:
                return ProductKernel(self.kernels + [other])
        else:
            raise ValueError("Can only multiply Kernel with another Kernel")

    def compute(self, x1, x2):
        return np.prod([kernel.compute(x1, x2) for kernel in self.kernels], axis=0)


class SumKernel(Kernel):
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels

    def __str__(self):
        return " + ".join(str(kernel) for kernel in self.kernels)

    def __add__(self, other):
        if isinstance(other, Kernel):
            if isinstance(other, SumKernel):
                return SumKernel(self.kernels + other.kernels)
            else:
                return SumKernel(self.kernels + [other])
        else:
            raise ValueError("Can only add Kernel with another Kernel")

    def compute(self, x1, x2):
        return np.sum([kernel.compute(x1, x2) for kernel in self.kernels], axis=0)
