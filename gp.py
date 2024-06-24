import numpy as np


class GaussianProcess:

    def __init__(self, X, Y, noise=1e-3, length_scale=1.0, amplitude=1.0):
        self.X = X
        self.Y = Y
        self.noise = noise
        self.length_scale = length_scale
        self.amplitude = amplitude

    def kernel(self, x1, x2):
        sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.amplitude**2 * np.exp(-0.5 / self.length_scale**2 * sq_dist)

    def predict(self, X_pred):
        K_inv = np.linalg.inv(self.kernel(self.X, self.X) + self.noise * np.eye(len(self.X)))
        K_star = self.kernel(self.X, X_pred)
        K_star_star = self.kernel(X_pred, X_pred)

        mu = K_star.T.dot(K_inv).dot(self.Y)
        cov = K_star_star - K_star.T.dot(K_inv).dot(K_star)

        return mu.flatten(), np.diag(cov)

    def update_data(self, X_new, Y_new):
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

