from BaseGradDesc import BaseGradDesc
import numpy as np
import matplotlib.pyplot as plot

__author__ = 'Lucas Ramadan'

class NAGD(BaseGradDesc):
    def __init__(self, n_steps=50, step_size=20, beta=1.0):
        BaseGradDesc.__init__(self, n_steps, step_size)
        self.lamb = 0.0
        self.beta = beta # Lipschitz Constant

    def _update_weights(self, gradient):
        next_lambda = (1 + (1 + 4 * self.lamb ** 2) ** 0.5) / 2
        gamma = (1 - self.lamb) / next_lambda
        self.lamb = next_lambda

        y_vec = self.weights - (1.0 / self.beta) * gradient
        self.weights = (1 - gamma) * y_vec + y_vec * gamma
