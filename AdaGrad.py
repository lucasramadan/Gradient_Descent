from BaseGradDesc import BaseGradDesc
import numpy as np
import matplotlib.pyplot as plot

__author__ = 'Lucas Ramadan'

class AdaGrad(BaseGradDesc):
    def __init__(self, n_steps=50, step_size=20, eta=1.0, epsilon=1e-3):
        BaseGradDesc.__init__(self, n_steps, step_size)
        self.eta = eta # numerator of step expression
        self.epsilon = epsilon # smoothing constant

    def _get_step(self, gradient):
        return self.eta / ((self.sq_gradient_sum + self.epsilon)**0.5) * gradient

    def _update_weights(self, gradient):
        self.weights = self.weights - self._get_step(gradient)

    def fit(self, X, Y):

        # save data and labels for plotting methods
        self.data = X
        self.labels = Y

        # get number of observations, features from data
        n_obs, n_features = X.shape

        # now make the weights attribute
        self.weights = np.random.rand(n_features)
        self.weights_history.append(self.weights)

        # make a sq_gradient_sum, to hold the running total of squared gradients
        self.sq_gradient_sum = np.zeros(n_features)

        # make n_steps number of steps
        for _ in xrange(self.n_steps):

            # start with grad, cost as zeros
            gradient = np.zeros(n_features)
            cost = 0.0

            # calculate gradient and cost
            for y_i, x_i in zip(Y, X):
                error = y_i - self._sigmoid(x_i)
                cost += 0.5 * error**2 / n_obs
                gradient += -error * self._sigmoid(x_i) * self._sigmoid(-1.*x_i) * x_i / n_obs

            # add the gradient to the running sum
            self.sq_gradient_sum += gradient**2

            # update the weights, with a step in the direction of the gradient
            self._update_weights(gradient)

            # keep track of changes in weights, cost, gradient, steps
            self.steps_history.append(self._get_step(gradient))
            self.weights_history.append(self.weights)
            self.cost_history.append(cost)
            self.gradient_history.append(gradient)

        # finally, convert the lists to np arrays, for ease of plotting
        self.weights_history = np.asarray(self.weights_history)
        self.cost_history = np.asarray(self.cost_history)
        self.steps_history = np.asarray(self.steps_history)
        self.gradient_history = np.asarray(self.gradient_history)
