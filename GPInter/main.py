import numpy as np
from gp import GaussianProcess, RBFKernel, PeriodicKernel, RationalQuadraticKernel
from plot import plot_gp

np.random.seed(1840319)


X_init = np.arange(-5, 5, 0.5).reshape(-1, 1)
Y_init = np.sin(X_init)
Y_err = np.random.normal(0, 0.1, size=(X_init.shape))
# Y_err = np.zeros(shape=(X_init.shape))


X_pred = np.arange(-5, 5, 0.1).reshape(-1, 1)

kernel1 = RBFKernel(length_scale=1.0, amplitude=1.0)
kernel2 = PeriodicKernel(length_scale=1.0, amplitude=1.0)
kernel3 = RationalQuadraticKernel(length_scale=1.0, amplitude=1.0)

gp1 = GaussianProcess(X_init, Y_init, Y_err=Y_err, kernel=kernel1)
gp2 = GaussianProcess(X_init, Y_init, Y_err=Y_err, kernel=kernel2)
gp3 = GaussianProcess(X_init, Y_init, Y_err=Y_err, kernel=kernel3)

print(gp1.kernel)
print(gp2.kernel)
print(gp3.kernel)
plot_gp([gp1, gp2, gp3], X_pred)
# plot_gp(gp3, X_pred)
# plot_gp(gp3, X_pred)
