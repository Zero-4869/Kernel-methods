import numpy as np
from scipy import optimize


class KernelSVC:
    def __init__(self, C, kernel, epsilon=1e-3):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

        self.Nfeval = 0
    def fit(self, X, y):
        y = 2 * (y - 0.5)

        N = len(y)
        self.support = X

        M = self.kernel(X, X)
        # Lagrange dual problem
        def loss(alpha):
            Y = np.diag(y)
            AY = Y @ alpha
            return -np.sum(alpha) + AY.T @ (M @ AY) / 2

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            Y = np.diag(y)
            AY = Y @ alpha
            return -np.ones(len(alpha)) + Y @ (M @ AY)

        fun_eq = lambda alpha: np.sum(alpha * y)
        jac_eq = lambda alpha: y
        fun_ineq1 = lambda alpha: self.C - alpha
        jac_ineq1 = lambda alpha: -np.identity(len(alpha))
        fun_ineq2 = lambda alpha: alpha
        jac_ineq2 = lambda alpha: np.identity(len(alpha))

        constraints = ({'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 'fun': fun_ineq1, 'jac': jac_ineq1},
                       {'type': 'ineq', 'fun': fun_ineq2, 'jac': jac_ineq2})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints,
                                   callback=self.callbackF,
                                   options={"maxiter":50, 'disp':True})
        self.alpha = optRes.x
        ## Assign the required attributes
        Y = np.diag(y)
        AY = Y @ self.alpha

        self.margin_points = [X[p] for p in np.where((self.alpha > self.epsilon) & (self.alpha < self.C - self.epsilon))[0]]
        self.b = np.mean(y[np.where((self.alpha > self.epsilon) & (self.alpha < self.C - self.epsilon))[0]]
                         - self.kernel(self.margin_points,X) @ AY)
        self.norm_f = np.sqrt(AY.T @ M @ AY)
        self.alpha = AY

    ### Implementation of the separting function $f$
    def separating_function(self, x):
        return self.kernel(x, self.support) @ self.alpha

    def predict(self, X):
        d = self.separating_function(X)
        # return 2 * (d + self.b > 0) - 1
        return 1 / (1 + np.exp(-(d+self.b)))

    def callbackF(self, Xi):
        print(self.Nfeval)
        self.Nfeval += 1

