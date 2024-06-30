import numpy as np


class GaussianProcess:
    def __init__(self, X, Y, Y_err, noise=1e-3, kernel=None):
        self.X = X
        self.Y = Y
        self.Y_err = Y_err.flatten()
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

    def update_data(self, X_new, Y_new, Y_err_new):
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.Y_err = np.concatenate((self.Y_err, Y_err_new.flatten()))
