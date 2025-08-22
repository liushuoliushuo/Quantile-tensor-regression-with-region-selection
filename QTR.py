import matplotlib.pyplot as plt
import numpy as np
import tensorly
from tensorly.base import partial_tensor_to_vec, partial_unfold
from tensorly.cp_tensor import cp_to_tensor
from tensorly import kron
import copy
import warnings
warnings.filterwarnings("ignore")

class QTR:
    def __init__(self,
                 tau = 0.5,
                 rank = 2,
                 max_iter = 100,
                 tol = 1e-7,
                 model_lambda = 0.1,
                 admm_varrho = 0.1,
                 penalty = 'scad',
                 a = 3.7,
                 B_init = None):

        self.tau = tau
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.model_lambda = model_lambda
        self.admm_varrho = admm_varrho
        self.penalty = penalty
        self.a = a
        self.B_init = B_init

    def fit(self, y, x):

        beta = np.zeros(shape = x.shape[1:])

        for it in range(10):

            if self.penalty == 'scad':
                weight = self.scad_derivative(np.abs(beta), self.model_lambda) / self.model_lambda
            elif self.penalty == 'mcp':
                weight = self.mcp_derivative(np.abs(beta), self.model_lambda) / self.model_lambda
            else:
                raise ValueError("Error in selection of penalty function")

            model = self.weight_regression(y, x, weight)
            new_beta = self.W

            if np.linalg.norm(new_beta - beta) / np.linalg.norm(beta) < 1e-7:
                beta = new_beta
                break

            beta = new_beta

        return beta


    def weight_regression(self, y, x, weight):

        self.n = len(y)
        self.y = y.reshape(self.n, 1)
        self.x = x
        self.modes = tensorly.ndim(x) - 1
        self.weight = weight

        rng = tensorly.check_random_state(None)

        if self.B_init == None:
            self.B = []
            for i in range(1, tensorly.ndim(x)):
                self.B.append(tensorly.tensor(rng.randn(self.x.shape[i], self.rank), **tensorly.context(self.x)))
        else:
            self.B = self.B_init

        weights = tensorly.ones(self.rank, **tensorly.context(self.x))
        self.coef_tensor = cp_to_tensor((weights, self.B))

        self.z = copy.deepcopy(self.y) - tensorly.tenalg.inner(self.x, self.coef_tensor, self.modes).reshape(self.n, 1)
        self.W = self.coef_tensor
        self.xi = np.zeros_like(self.z)
        self.G = np.zeros_like(self.W)

        for it in range(self.max_iter):

            W_old = self.W

            for m in range(self.modes):

                B_m = tensorly.tenalg.khatri_rao(self.B, skip_matrix=m)
                B_m_extend = kron(B_m, np.eye(self.B[m].shape[0]))
                D = tensorly.reshape(tensorly.dot(partial_unfold(self.x, m, skip_begin=1), B_m), (self.n, -1))
                inverse_term = np.linalg.inv(np.dot(D.T, D) + tensorly.dot(B_m_extend.T, B_m_extend))

                other_term = D.T.dot(self.xi) + self.admm_varrho*D.T.dot(self.y)-self.admm_varrho*D.T.dot(self.z) - np.dot(partial_unfold(self.G, m, skip_begin=0), B_m).reshape(-1, 1)+self.admm_varrho*B_m_extend.T.dot(partial_unfold(self.W, m, skip_begin=0).reshape(-1, 1))

                tmp = (1 / self.admm_varrho) * np.dot(inverse_term, other_term)
                self.B[m] = tmp.reshape(self.B[m].shape[0], self.rank)

            self.coef_tensor = cp_to_tensor((weights, self.B))

            self.z = self.prox_check_function(copy.deepcopy(self.y) - tensorly.tenalg.inner(self.x, self.coef_tensor, self.modes).reshape(self.n, 1) + self.xi / self.admm_varrho, self.tau,
                                         self.admm_varrho)

            self.W = self.prox_lasso(self.coef_tensor + self.G/self.admm_varrho, self.model_lambda * self.weight / self.admm_varrho)


            self.xi = self.xi - self.admm_varrho*(self.z - self.y + tensorly.tenalg.inner(self.x, self.coef_tensor, self.modes).reshape(self.n, 1))
            self.G = self.G - self.admm_varrho * (self.W - self.coef_tensor)

            if np.linalg.norm(self.W - W_old)/ np.linalg.norm(W_old) < 1e-7:
                break

    def scad_derivative(self, beta, lambda_val, a=3.7):

        abs_beta = np.abs(beta)
        return np.where(
            abs_beta <= lambda_val,
            lambda_val,
            np.where(
                abs_beta <= a * lambda_val,
                (a * lambda_val - abs_beta) / (a - 1),
                0
            )
        )

    def mcp_derivative(self, beta, lambda_val, a = 3.7):
        abs_beta = np.abs(beta)
        return np.where(
            abs_beta <= a * lambda_val,
            (lambda_val - abs_beta/a) * np.sign(beta),
            0
        )

    def prox_check_function(self, v, tau, alpha):
        return v - np.maximum((tau - 1) / alpha, np.minimum(v, tau / alpha))

    def prox_lasso(self, v, a):
        return np.maximum(v - a, 0) - np.maximum(-v - a, 0)

    def check_function(self, residuals, tau):
        loss = np.where(residuals >= 0, tau * residuals, (tau - 1) * residuals)
        return np.sum(loss)

    def BIC(self):
        loss = self.check_function(self.y - tensorly.dot(partial_tensor_to_vec(self.x), self.W.reshape(-1, 1)), self.tau)

        df = np.sum(self.W != 0)
        bic = np.log(loss / self.n) + (np.log(self.n) * (df)) / (2 * self.n)
        return bic

    def predict(self, X):
        out = tensorly.dot(partial_tensor_to_vec(X), self.W.reshape(-1, 1))
        return out
