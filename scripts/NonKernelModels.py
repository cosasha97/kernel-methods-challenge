from scipy import optimize
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np


'''
Logistic Regression
'''


def g(x):
    return 1 / (1 + np.exp(-x))


def f(z, X, Y, _lambda):
    """
    Define the function at which we want to find the zeros
    """
    w = z[:-1]
    beta = np.squeeze(z[-1])
    term_1_d = (X * (Y[:, np.newaxis] - g(X.dot(w) + beta))).sum(axis=0)
    term_d_1 = (Y[:, np.newaxis] - g(X.dot(w) + beta)).sum() + 2 * _lambda * beta
    return np.hstack((term_1_d, term_d_1))[:, np.newaxis]


def hess_f(z, X, Y, _lambda):
    """
    Compute the Hessian of the function f
    """
    d = z.shape[0]
    w = z[:-1]
    beta = z[-1]
    hess = np.zeros((d, d))
    hess[:-1, :-1] = - np.einsum('ki,kj->ij', X * g(X.dot(w) + beta), X * g(-(X.dot(w) + beta)))
    hess[:-1, [-1]] = - np.einsum('ij,ik->kj', g(X.dot(w) + beta) * g(-(X.dot(w) + beta)), X) + 2 * _lambda
    hess[[-1], :-1] = hess[:-1, [-1]].T
    hess[-1, -1] = - np.dot(g(X.dot(w) + beta).T, g(-(X.dot(w) + beta))) + 2 * _lambda
    return hess


def NewtonRaphson(z_ini, f, fprime, tol):
    """
    Create Newton Raphson algorithm to find the zeros of a fonction f.
    Args :
        z_ini : starting point of the algorithm
        f : function f
        f_prime : hessian of f
        tol : tolerance
    Returns :
        z_current : approximative zero of f
    """
    z_current = z_ini
    while np.linalg.norm(f(z_current)) > tol:
        hess = fprime(z_current)
        inv_hess = np.linalg.solve(hess, np.eye(hess.shape[0]))
        z_new = z_current - inv_hess.dot(f(z_current))
        z_current = z_new
    return z_current


### Logisitc Regression Class
class LogisticRegression:
    def __init__(self, _lambda=0):
        self._lambda = 0
        pass

    def fit(self, X, Y):
        n, p = X.shape
        z_ini = np.zeros((p + 1, 1))
        z_final = NewtonRaphson(z_ini, f=lambda z: f(z, X, Y, self._lambda),
                                fprime=lambda z: hess_f(z, X, Y, self._lambda), tol=1e-8)
        w_final, beta_final = z_final[:-1], z_final[-1]
        self.w = w_final
        self.beta = beta_final

    def predict(self, X):
        proba_success = g(X.dot(self.w) + self.beta)
        label_successes = (proba_success > 0.5)
        return label_successes


'''
Ridge Regression
'''


class RidgeRegression:

    def __init__(self, lambda_=0.05):
        self.lambda_ = lambda_

    def fit(self, df_train, df_label):
        X_train = np.array(df_train)
        Y_train = np.array(df_label).reshape(-1, 1)
        Y_train = -1 + 2 * Y_train

        A = np.multiply(X_train[:, :, np.newaxis], X_train[:, np.newaxis, :]).sum(axis=0) + self.lambda_ * np.eye(
            X_train.shape[1])
        b = (Y_train * X_train).sum(axis=0).reshape(-1, 1)
        self.theta = np.linalg.solve(A, b)

    def predict(self, df):
        X = np.array(df)
        Y = np.sign(np.dot(X, self.theta))
        return ((Y + 1) // 2).astype(np.int)

    def get_params(self, deep=False):
        return {'lambda_': self.lambda_}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self