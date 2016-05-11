import numpy as np
import matplotlib.pyplot as plot

__author__ = "Lucas Ramadan"

class BaseGradDesc():
    def __init__(self, n_steps=50, step_size=20.0):
        self.n_steps = n_steps
        self.step_size = step_size
        self.weights_history = []
        self.cost_history = []
        self.steps_history = []
        self.gradient_history = []

    def _error(self, x, y):
        return y - np.dot(self.weights, x)

    def _Error(self, X, y):
        """

        """
        return

    def _cost(self):
        """
        MSE calculation, given
        """
        return 0.5 / self.n_obs * np.sum(np.subtract(self.labels,
                                            self._Sigmoid(self.data))**2)

    def _sigmoid(self, x):
        """
        Return the value of the sigmoid function for an observation vector x
        """
        return 1.0 / (1.0 + np.exp(-(np.dot(self.weights, x))))

    def _Sigmoid(self, X):
        """
        Return the vector of predictions from observations in matrix X
        """
        return np.asarray([self._sigmoid(x) for x in X])

    def _update_weights(self, gradient):
        self.weights = self.weights - self._get_step(gradient)

    def _get_step(self, gradient):
        return self.step_size * gradient

    def fit(self, X, Y):

        # save data and labels for plotting methods
        self.data = X
        self.labels = Y

        # get number of observations, features from data
        self.self.n_obs, self.n_features = X.shape

        # now make the weights attribute
        self.weights = np.random.rand(self.n_features)
        self.weights_history.append(self.weights)

        # make n_steps number of steps
        for _ in xrange(self.n_steps):

            # start with grad, cost as zeros
            gradient = np.zeros(self.n_features)

            # refactored, optimzed code with numpy
            cost = self._cost()

            # calculate gradient and cost
            for y_i, x_i in zip(Y, X):
                error = y_i - self._sigmoid(x_i)
                gradient += -error * self._sigmoid(x_i) * self._sigmoid(-1.*x_i) * x_i / self.n_obs

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

    def generate_data(self, n_features=2, n_obs=200, noise=0.2, intercept=False):
        # whether or not we are interested in solving for an intercept (b) term
        self.intercept = intercept

        # generate the random data
        X = np.random.rand(self.n_obs, self.n_features)
        Y = np.asarray([1.0 if x2 > (x1 + np.random.normal(0.0, noise)) else 0.0 for x1, x2 in X])

        # if want to calculate an intercept term, add column of ones to X
        # XXX: this means that the first weight IS the intercept term (b)
        if intercept:
            X = np.append(np.ones((data.shape[0], 1)), X, axis=1)

        return X, Y

    def plot_data(self):

        colorMap = {0.0: "blue", 1.0: "red"}
        colors = [colorMap[c] for c in self.labels]

        plot.scatter(self.data[:, 0], self.data[:,1] , c=colors)
        plot.xlabel('x1')
        plot.ylabel('x2')
        plot.show()

    def plot_weights(self):
        plot.plot(xrange(self.weights_history.shape[0]), self.weights_history[:, 0])
        plot.plot(xrange(self.weights_history.shape[0]), self.weights_history[:, 1])
        plot.xlabel('Number of Gradient Descent Steps')
        plot.ylabel('Network Weights')
        plot.show()

    def plot_performance(self):
        j2 = [2*x for x in self.cost_history]
        plot.plot(xrange(len(self.cost_history)), j2)
        plot.xlabel('Number of Gradient Descent Steps')
        plot.ylabel('J Performance')
        plot.show()

    def plot_steps(self):
        plot.plot(self.steps_history[:,0], self.steps_history[:,1])
        plot.xlabel('Step 1')
        plot.ylabel('Step 2')
        plot.show()
