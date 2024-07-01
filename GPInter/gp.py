import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


class GaussianProcess:
    def __init__(self, X, Y, Y_err, noise=1e-3, kernel=None):
        self.X = X
        self.Y = Y
        self.Y_err = Y_err.flatten()
        self.noise = noise
        self.kernel = kernel
        self.init_params = self.kernel.get_params()
        self._neg_log_likelihood = self.neg_log_likelihood()

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

    def neg_log_likelihood(self, params=None):
        if params is not None:
            try:
                self.kernel.set_params(**dict(zip(self.kernel.get_params().keys(), params)))
            except ValueError as e:
                print(f"Invalid value for {params}: {e}. Re-setting to initial value = {self.init_params}.")
                self.kernel.set_params(**dict(zip(self.kernel.get_params().keys(), list(self.init_params.values()))))

        K = self.kernel.compute(self.X, self.X)
        K += np.diag(self.Y_err**2)
        K += self.noise * np.eye(len(self.X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as e:
            print(f"Optimization failed due to a LinAlgError: {e} while computing the Cholesky decomposition.")
            return None
        S1 = solve_triangular(L, self.Y, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)

        return np.sum(np.log(np.diag(L))) + 0.5 * self.Y.T.dot(S2) + 0.5 * len(self.X) * np.log(2 * np.pi)

    def optimize(self, set_optimized_params=False):

        if self._neg_log_likelihood is None:
            return None

        callback_params = []

        def _store_callback(x):
            callback_params.append(x.copy())

        initial_params = list(self.kernel.get_params().values())

        opt_result = minimize(fun=self.neg_log_likelihood, x0=initial_params, method="L-BFGS-B", callback=_store_callback)

        optimized_params = dict(zip(self.kernel.get_params().keys(), opt_result.x))

        if set_optimized_params:
            self.kernel.set_params(**optimized_params)

        self.callback_params = callback_params

        return optimized_params
