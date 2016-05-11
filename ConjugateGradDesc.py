from BaseGradDesc import BaseGradDesc
import numpy as np
import matplotlib.pyplot as plot
from scipy import optimize

__author__ = "Lucas Ramadan"

class ConjugateGD(BaseGradDesc):
    def __init__(self, n_steps=50, step_size=20.0):
        BaseGradDesc.__init__(self, n_steps, step_size)

    def cost(self, *args):
        cost = 0.0

        self.weights, X, Y = args

        for y_i, x_i in zip(Y, X):
            error = y_i - self._sigmoid(x_i)
            cost += 0.5 * error**2 / self.n_obs

        return cost

    def gradient(self, *args):
        gradient = np.zeros(self.n_features)

        self.weights, X, Y = args

        for y_i, x_i in zip(Y, X):
            error = y_i - self._sigmoid(x_i)
            gradient += -error * self._sigmoid(x_i) * self._sigmoid(-1.*x_i) * x_i / self.n_obs

        return gradient

    def fit(self, X, Y):
        # save data and labels for plotting methods
        self.data = X
        self.labels = Y

        # get number of observations, features from data
        self.n_obs, self.n_features = X.shape

        # now make the weights attribute
        self.weights = np.random.rand(self.n_features)
        self.weights_history.append(self.weights)

        # use the scipy optimize Conjugate Gradient method
        optimize.fmin_cg(self.cost, self.weights, fprime=self.gradient, args=(X, Y))
