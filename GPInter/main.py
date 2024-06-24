import numpy as np
from gp import GaussianProcess
from plot import plot_gp

np.random.seed(1840319)


X_init = np.random.uniform(-5, 5, size=(6, 1))
Y_init = np.random.uniform(-2, 2, size=(6, 1))
X_pred = np.arange(-5, 5, 0.1).reshape(-1, 1)

gp = GaussianProcess(X_init, Y_init, length_scale=1.0, amplitude=1)

plot_gp(gp, X_pred)