from BaseGradDesc import BaseGradDesc
import numpy as np
import matplotlib.pyplot as plot

__author__ = 'Lucas Ramadan'

class AdamGD(BaseGradDesc):
    def __init__(self, n_steps=50, step_size=20,
                    alpha=0.001, beta_m=0.9, beta_v=0.999, epsilon=1e-8):
        BaseGradDesc.__init__(self, n_steps, step_size)
        self.alpha = alpha
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.beta_m_t = 1.0
        self.beta_v_t = 1.0
        self.m = None
        self.v = None
        self.epsilon = epsilon

def _get_step(self, gradient):
    # calculate the gradient step for updating weights, based on m_hat and v_hat
    return self.alpha * self.m_hat / (self.v_hat**0.5 + self.epsilon)

def _update_weights(self, gradient):
    # first update the first and second moment vectors
    self.m = (self.beta_m * self.m) + ((1 - self.beta_m) * gradient)
    self.v = (self.beta_v * self.v) + ((1 - self.beta_v) * gradient**2)

    # now update beta_m_t and beta_v_t
    self.beta_m_t = self.beta_m_t*self.beta_m
    self.beta_v_t = self.beta_v_t * self.beta_v

    # next calculate m_hat and v_hat
    self.m_hat = self.m / (1 - self.beta_m_t)
    self.v_hat =self.v / (1 - self.beta_v_t)

    # finally, update the weights
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

    print self.weights

    # make vector of first moments (m) and vector of second moments (v)
    self.m = np.zeros(n_features)
    self.v = np.zeros(n_features)

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
